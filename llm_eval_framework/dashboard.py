"""
dashboard.py — Streamlit UI for the LLM Evaluation Framework

Run with:
    cd llm_eval_framework
    streamlit run dashboard.py
"""
import os
import sys
import json
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from config import RESULTS_FILE  # noqa: E402

st.set_page_config(page_title="LLM Eval Dashboard", layout="wide", page_icon="📊")
st.title("📊 Prompt-wise LLM Evaluation Framework")
st.caption("Atomic claim-level hallucination and faithfulness analysis for local RAG systems.")


@st.cache_data
def load_results(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── File selector ─────────────────────────────────────────────────────────────
results_path = RESULTS_FILE
if not os.path.exists(results_path):
    st.error(
        f"No results file found at `{results_path}`. "
        "Run `python pipeline.py --mini` first to generate results."
    )
    st.stop()

results = load_results(results_path)
prompt_data = results.get("prompts", [])

if not prompt_data:
    st.warning("Results file is empty. Run the pipeline first.")
    st.stop()

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Run Info")
    st.metric("Generator Model", results.get("generator_model", "—"))
    st.metric("Extractor Model", results.get("claim_extractor_model", "—"))
    st.metric("Fast Mode", str(results.get("fast_mode", "—")))
    st.metric("Prompts Evaluated", len(prompt_data))
    st.divider()
    st.header("Thresholds")
    st.caption("Edit config.py to change these.")
    st.metric("Faithfulness threshold", 0.5)
    st.metric("Confidence threshold", 0.4)

# ── Build summary dataframe ───────────────────────────────────────────────────
rows = []
for p in prompt_data:
    rows.append({
        "ID": p["id"],
        "Type": p.get("type", "—"),
        "FRANQ Score": p.get("prompt_score"),
        "Grounding": p.get("grounding_score"),
        "Hallucination Rate": p.get("hallucination_rate"),
        "Consistency": p.get("consistency_score"),
        "Calibration": p.get("calibration"),
        "Relevance": p.get("relevance_score"),
        "Completeness": p.get("completeness_score"),
        "Robustness": p.get("robustness_score"),
        "# Claims": len(p.get("claims", [])),
    })
df = pd.DataFrame(rows)

# ── Overall summary metrics ───────────────────────────────────────────────────
st.header("Overall Model Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Model Score", round(results.get("model_score", 0), 3))
c2.metric("Avg Hallucination Rate", round(df["Hallucination Rate"].mean(), 3))
c3.metric("Avg Grounding", round(df["Grounding"].mean(), 3))
c4.metric("Avg Consistency", round(df["Consistency"].mean(), 3))
c5.metric("Avg Relevance", round(df["Relevance"].mean(), 3))

# ── Per-type breakdown ────────────────────────────────────────────────────────
st.subheader("Scores by Prompt Type")
type_df = df.groupby("Type")[["FRANQ Score", "Grounding", "Hallucination Rate"]].mean().round(3)
st.dataframe(type_df, use_container_width=True)

# ── Per-prompt table ──────────────────────────────────────────────────────────
st.header("Per-Prompt Metrics")
st.dataframe(df, use_container_width=True, height=350)

# ── Score distribution chart ──────────────────────────────────────────────────
st.subheader("FRANQ Score Distribution by Prompt")
chart_df = df[["ID", "FRANQ Score"]].set_index("ID")
st.bar_chart(chart_df)

# ── Error taxonomy across all prompts ────────────────────────────────────────
st.header("Error Taxonomy — All Claims")
all_classifications = [
    c["classification"]
    for p in prompt_data
    for c in p.get("claims", [])
]
if all_classifications:
    taxonomy_df = (
        pd.Series(all_classifications)
        .value_counts()
        .reset_index()
    )
    taxonomy_df.columns = ["Category", "Count"]
    st.bar_chart(taxonomy_df.set_index("Category"))
    total = len(all_classifications)
    st.caption(f"Total claims evaluated: {total}")

# ── Prompt drill-down ─────────────────────────────────────────────────────────
st.header("Prompt Drill-Down")
selected_id = st.selectbox("Select a Prompt ID", [p["id"] for p in prompt_data])
selected = next((p for p in prompt_data if p["id"] == selected_id), None)

if selected:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Prompt")
        st.info(selected["prompt"])
        st.caption(f"Type: **{selected.get('type', '—')}**")

    with col_right:
        st.subheader("Response")
        st.write(selected.get("response", "—"))

    st.subheader("Retrieved Context Documents")
    for i, doc in enumerate(selected.get("context", []), 1):
        with st.expander(f"Context Doc {i}"):
            st.write(doc[:600] + ("..." if len(doc) > 600 else ""))

    st.subheader("Extracted Claims")
    claims = selected.get("claims", [])
    if claims:
        claim_rows = []
        for c in claims:
            claim_rows.append({
                "Claim": c["text"],
                "Faithfulness": c.get("faithfulness"),
                "Confidence": c.get("confidence"),
                "Parametric": c.get("parametric"),
                "Contradiction": c.get("contradiction"),
                "FRANQ Score": c.get("franq_score"),
                "Classification": c.get("classification"),
            })
        claim_df = pd.DataFrame(claim_rows)
        st.dataframe(claim_df, use_container_width=True)
    else:
        st.warning("No claims extracted for this prompt.")

    st.subheader("Prompt-Level Metric Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("FRANQ Score", selected.get("prompt_score", "—"))
    m2.metric("Hallucination Rate", selected.get("hallucination_rate", "—"))
    m3.metric("Grounding Score", selected.get("grounding_score", "—"))
    m4.metric("Calibration", selected.get("calibration", "—"))
