"""Demo: uv run streamlit run src/dve/app/streamlit_app.py"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dve.app import inference as inf

st.set_page_config(page_title="DeepVariantEffect")
st.title("DeepVariantEffect")
st.warning("A portfolio project. Do not use it for clinical decisions.")


@st.cache_resource
def resources():
    return inf.load_model(), inf.platt(), pd.read_parquet("reports/app_variants.parquet")


model, (a, b), table = resources()
st.write(
    "Scores a missense variant from the 1,025 bp of DNA around it, using the project's CNN. "
    "The listed variants come from held-out chromosomes (8 and 21)."
)

mode = st.radio("Input", ["Held-out ClinVar variant", "Paste a sequence"], horizontal=True)
row = None
if mode.startswith("Held"):
    gene = st.selectbox("Gene", sorted(table["gene"].unique()))
    sub = table[table["gene"] == gene].sort_values("pos")
    names = sub["chrom"] + ":" + sub["pos"].astype(str) + " " + sub["ref"] + ">" + sub["alt"]
    row = sub.iloc[list(names).index(st.selectbox("Variant", names))]
    ref_seq, alt_seq = row["ref_seq"], row["alt_seq"]
else:
    ref_seq = st.text_area(f"Reference window ({inf.WINDOW} bases, variant in the middle)")
    ref_seq = "".join(ref_seq.upper().split())
    alt = st.selectbox("Alternate base", list("ACGT"))
    error = inf.validate_window(ref_seq) if ref_seq else "Paste a sequence to score it."
    if not error and ref_seq[inf.MID] == alt:
        error = "The alternate base equals the reference base."
    if error:
        st.info(error)
        st.stop()
    alt_seq = inf.make_alt_seq(ref_seq, alt)

logit = inf.logits(model, [ref_seq], [alt_seq])[0]
st.metric("Predicted probability of being pathogenic", f"{inf.probability(logit, a, b):.2f}")
if row is not None:
    st.caption(f"ClinVar: {row['clnsig']}, {row['stars']} review star(s).")

st.subheader("What drives the score")
st.caption("Change in the score when a nearby base is changed. Blue lowers it, red raises it.")
span = 30
sal = inf.saliency(model, ref_seq, alt_seq, span)
limit = float(np.abs(sal).max()) or 1.0
fig = go.Figure(
    go.Heatmap(
        z=sal,
        x=list(range(-span, span + 1)),
        y=list("ACGT"),
        zmin=-limit,
        zmax=limit,
        zmid=0,
        colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]],
        colorbar={"title": "Change"},
    )
)
fig.update_layout(
    xaxis_title="Distance from the variant (bases)",
    height=260,
    margin={"l": 10, "r": 10, "t": 10, "b": 10},
)
st.plotly_chart(fig)

if row is not None:
    st.subheader("Nearest ClinVar variants in this gene")
    near = sub[sub["variation_id"] != row["variation_id"]].copy()
    near["distance"] = (near["pos"] - row["pos"]).abs()
    cols = ["pos", "ref", "alt", "clnsig", "stars", "distance"]
    st.dataframe(near.nsmallest(8, "distance")[cols], hide_index=True)
