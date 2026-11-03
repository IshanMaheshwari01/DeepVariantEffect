"""Model and evaluation tests. Torch tests are skipped in CI, which installs no torch."""

import numpy as np
import pytest

from dve.eval import external as ext
from dve.eval import stats as M
from dve.features import encode


def test_gene_bootstrap_is_wider_when_genes_are_correlated():
    rng = np.random.default_rng(0)
    genes = np.repeat(np.arange(40), 25)
    y = rng.integers(0, 2, 1000)
    s = y * 0.5 + np.repeat(rng.normal(size=40), 25) + rng.normal(size=1000) * 0.1
    lo, hi = M.bootstrap_ci(y, s, genes, n_boot=100)
    lo2, hi2 = M.bootstrap_ci(y, s, np.arange(1000), n_boot=100)
    assert (hi - lo) > (hi2 - lo2)


def test_within_gene_auroc_ignores_gene_level_signal():
    y = np.array([0] * 6 + [1] * 6 + [0] * 6 + [1] * 6)
    s = np.r_[np.arange(12.0), np.arange(12.0) + 100]
    assert M.within_gene_auroc(y, s, ["a"] * 12 + ["b"] * 12) == (1.0, 2)


def test_calibration_helpers():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    assert M.ece(y, p) < 0.03
    assert np.isfinite(M.platt_calibrate(y, p, p)).all()


def test_parse_myvariant_hit():
    # shape returned by the live API for chr7:g.140753336A>T
    hit = {
        "query": "chr7:g.140753336A>T",
        "dbnsfp": {
            "alphamissense": {"score": [0.98, 0.99, 0.97]},
            "cadd": {"phred": 29.8},
            "revel": {"score": 0.931},
        },
    }
    out = ext.parse_hit(hit)
    assert out == {
        "hgvs": "chr7:g.140753336A>T",
        "cadd_phred": 29.8,
        "revel": 0.931,
        "alphamissense": 0.99,
    }
    assert ext.parse_hit({"query": "x", "dbnsfp": {}})["revel"] != 0.0
    assert ext.parse_hit({"query": "x", "cadd": {"phred": 12.0}})["cadd_phred"] == 12.0


def test_fetch_batch_skips_notfound():
    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"query": "a", "notfound": True}, {"query": "b", "dbnsfp": {}}]

    class S:
        def post(self, *a, **k):
            assert k["params"] == {"assembly": "hg38"}
            return R()

    assert [h["hgvs"] for h in ext.fetch_batch(["a", "b"], S())] == ["b"]


def test_report_refuses_smoke(tmp_path, monkeypatch):
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location("mr", "scripts/make_report.py")
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)
    monkeypatch.setattr(mr, "RES", tmp_path)
    (tmp_path / "benchmark.json").write_text(json.dumps({"models": ["nt_lora"]}))
    (tmp_path / "nt_lora.json").write_text(json.dumps({"smoke": True}))
    with pytest.raises(SystemExit):
        mr.load_real(False)


def test_tokenizer_matches_esm_tokenizer(tmp_path):
    pytest.importorskip("torch")
    from itertools import product

    from transformers import EsmTokenizer

    from dve.models import nt

    # same order as the model's vocab.txt
    kmers = ["".join(p) for p in product("ATCG", repeat=6)]
    vocab = ["<unk>", "<pad>", "<mask>", "<cls>", *kmers, "A", "T", "C", "G", "N", "<eos>", "<bos>"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = EsmTokenizer(str(tmp_path / "vocab.txt"), eos_token=None)
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 1025)) for _ in range(3)]
    assert tok(seqs)["input_ids"] == nt.tokenize(encode(seqs)).tolist()


def test_variant_stays_in_one_token_on_both_strands():
    pytest.importorskip("torch")
    from dve.models import nt
    from dve.models.nt import revcomp

    rng = np.random.default_rng(0)
    r = rng.integers(0, 4, (1, 1025))
    a = r.copy()
    a[0, 512] = (a[0, 512] + 1) % 4
    for x, y in ((r, a), (revcomp(r), revcomp(a))):
        assert np.flatnonzero(nt.tokenize(x)[0] != nt.tokenize(y)[0]).tolist() == [86]


def test_nt_forward():
    torch = pytest.importorskip("torch")
    from dve.models import nt

    x = torch.randint(0, 4, (2, 1025))
    m = nt.build(tiny=True, lora_r=2)
    ids = torch.from_numpy(nt.tokenize(x.numpy()))
    m(ids, ids).sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    assert not any("embeddings" in k for k in nt.trainable_state(m))


def test_saliency_shape():
    torch = pytest.importorskip("torch")
    from dve.app import inference as inf
    from dve.models.cnn import VariantCNN

    torch.manual_seed(0)
    s = "".join(np.random.default_rng(0).choice(list("ACGT"), 1025))
    alt = "A" if s[512] != "A" else "C"
    model = VariantCNN(channels=8).eval()
    assert inf.saliency(model, s, inf.make_alt_seq(s, alt), 3).shape == (4, 7)
