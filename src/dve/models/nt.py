"""Nucleotide Transformer with LoRA adapters and a ref/alt classification head.

The model tokenises left to right into non-overlapping 6-mers, then single bases for the
remainder. In a 1,025 bp window the variant (base index 512) is therefore always in token
1 + 512 // 6 = 86 (token 0 is <cls>), for the forward and the reverse-complement strand.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
CLS_ID, KMER_START, SINGLE_START = 3, 4, 4100
_ATCG = np.array([0, 2, 3, 1])  # dve encoding A,C,G,T=0..3 -> vocab order A,T,C,G


def tokenize(x: np.ndarray, k: int = 6) -> np.ndarray:
    """Encoded sequences (n, L) -> token ids (n, 1 + L // k + L % k). Same ids as the
    Hugging Face tokenizer (train_nt.py checks this against the real vocab at start-up)."""
    v = _ATCG[x].astype(np.int64)
    n, length = v.shape
    n_kmers = length // k
    weights = 4 ** np.arange(k - 1, -1, -1)
    kmers = (v[:, : n_kmers * k].reshape(n, n_kmers, k) * weights).sum(2) + KMER_START
    rest = v[:, n_kmers * k :] + SINGLE_START
    return np.hstack([np.full((n, 1), CLS_ID), kmers, rest])


def revcomp(x: np.ndarray) -> np.ndarray:
    """Reverse complement of an array from dve.features.encode (A+T = C+G = 3)."""
    return 3 - x[:, ::-1]


def variant_token(window: int = 1025, k: int = 6) -> int:
    return 1 + (window // 2) // k


class NTVariantClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden: int, center: int, dropout: float = 0.1):
        super().__init__()
        self.backbone, self.center = backbone, center
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, ref: torch.Tensor, alt: torch.Tensor) -> torch.Tensor:
        """Token ids (batch, tokens) -> logits (batch,)."""
        h = self.backbone(input_ids=torch.cat([ref, alt])).last_hidden_state
        r, a = h.chunk(2)
        c = self.center
        feats = torch.cat(
            [r[:, c], a[:, c] - r[:, c], r[:, 1:].mean(1), (a - r)[:, 1:].mean(1)], dim=1
        )
        return self.head(feats).squeeze(1)


def build(
    tiny: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    grad_checkpointing: bool = True,
    window: int = 1025,
) -> NTVariantClassifier:
    """Load the backbone, freeze it and add LoRA. `tiny` builds a small random model with
    the same architecture, for testing without a download. lora_r=0 trains the head only."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, EsmConfig, EsmModel

    if tiny:
        cfg = EsmConfig(
            vocab_size=4105,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            pad_token_id=1,
            mask_token_id=2,
            max_position_embeddings=1002,
            position_embedding_type="absolute",
            token_dropout=True,
        )
        base = EsmModel(cfg, add_pooling_layer=False)
    else:
        base = AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False)
    hidden = base.config.hidden_size
    for p in base.parameters():
        p.requires_grad = False
    if lora_r > 0:
        if grad_checkpointing:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            base.enable_input_require_grads()
        base = get_peft_model(
            base,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["query", "value"],
            ),
        )
    return NTVariantClassifier(base, hidden, variant_token(window))


def trainable_state(model: nn.Module) -> dict:
    """Only the LoRA and head weights, so a checkpoint is megabytes rather than 2 GB."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.cpu() for k, v in model.state_dict().items() if k in keep}
