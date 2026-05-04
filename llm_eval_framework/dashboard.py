"""
dashboard.py — Streamlit UI for the LLM Evaluation Framework

Run with:
    cd llm_eval_framework
    streamlit run dashboard.py
"""
import os
import sys
import json
import glob
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

st.set_page_config(page_title="LLM Eval Dashboard", layout="wide", page_icon="📊")
st.title("📊 Prompt-wise LLM Evaluation Framework")
st.caption("Atomic claim-level hallucination and faithfulness analysis for local RAG systems.")


@st.cache_data
def load_results(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_results() -> list[str]:
    """Return sorted list of all outputs/results*.json files."""
    pattern = os.path.join(BASE_DIR, "outputs", "results*.json")
    files = sorted(glob.glob(pattern))
    return files


def short_label(path: str) -> str:
    return os.path.basename(path).replace(".json", "")


def build_df(prompt_data: list) -> pd.DataFrame:
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
            "# Claims": len(p.get("claims", [])),
        })
    return pd.DataFrame(rows)


# ── Discover all result files ─────────────────────────────────────────────────
all_result_files = discover_results()
if not all_result_files:
    st.error(
        "No results files found in `outputs/`. "
        "Run `python pipeline.py --mini` first to generate results."
    )
    st.stop()

all_labels = [short_label(f) for f in all_result_files]
label_to_path = dict(zip(all_labels, all_result_files))

# ── Sidebar: file selector + mode toggle ─────────────────────────────────────
with st.sidebar:
    st.header("Results File")
    selected_label = st.selectbox(
        "Select run to inspect",
        all_labels,
        index=len(all_labels) - 1,  # default to latest
    )
    selected_path = label_to_path[selected_label]

    st.divider()
    st.header("Compare Mode")
    compare_mode = st.toggle("Compare two runs", value=False)
    if compare_mode:
        compare_label = st.selectbox(
            "Second run to compare",
            [l for l in all_labels if l != selected_label],
        )
        compare_path = label_to_path.get(compare_label)
    else:
        compare_label = None
        compare_path = None

    st.divider()
    st.header("Thresholds")
    st.caption("Edit config.py to change these.")
    st.metric("Faithfulness threshold", 0.5)
    st.metric("Confidence threshold", 0.4)

# ── Load primary results ───────────────────────────────────────────────────────
results = load_results(selected_path)
prompt_data = results.get("prompts", [])

if not prompt_data:
    st.warning("Selected results file is empty. Run the pipeline first.")
    st.stop()

# ── Sidebar: run metadata ─────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.header("Run Info")
    st.metric("Generator Model", results.get("generator_model", "—"))
    st.metric("Extractor Model", results.get("claim_extractor_model", "—"))
    st.metric("Fast Mode", str(results.get("fast_mode", "—")))
    st.metric("Prompts Evaluated", len(prompt_data))

df = build_df(prompt_data)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON MODE
# ═══════════════════════════════════════════════════════════════════════════════
if compare_mode and compare_path:
    results2 = load_results(compare_path)
    prompt_data2 = results2.get("prompts", [])
    df2 = build_df(prompt_data2)

    st.header(f"Model Comparison: `{selected_label}` vs `{compare_label}`")

    # Side-by-side top-level metrics
    col_a, col_b = st.columns(2)
    numeric_cols = ["FRANQ Score", "Grounding", "Hallucination Rate", "Consistency", "Relevance"]

    with col_a:
        st.subheader(f"🔵 {selected_label}")
        st.metric("Model Score", round(results.get("model_score", 0), 3))
        st.metric("Generator", results.get("generator_model", "—"))
        for col in numeric_cols:
            st.metric(f"Avg {col}", round(df[col].mean(), 3))

    with col_b:
        st.subheader(f"🟠 {compare_label}")
        score_a = results.get("model_score", 0)
        score_b = results2.get("model_score", 0)
        st.metric("Model Score", round(score_b, 3), delta=round(score_b - score_a, 3))
        st.metric("Generator", results2.get("generator_model", "—"))
        for col in numeric_cols:
            delta = round(df2[col].mean() - df[col].mean(), 3)
            st.metric(f"Avg {col}", round(df2[col].mean(), 3), delta=delta)

    st.divider()

    # Per-prompt FRANQ comparison chart
    st.subheader("FRANQ Score per Prompt")
    common_ids = [p["id"] for p in prompt_data if any(p2["id"] == p["id"] for p2 in prompt_data2)]
    chart_data = pd.DataFrame({
        selected_label: {p["id"]: p.get("prompt_score", 0) for p in prompt_data},
        compare_label:  {p["id"]: p.get("prompt_score", 0) for p in prompt_data2},
    }).loc[common_ids]
    st.bar_chart(chart_data)

    # Per-prompt grounding comparison chart
    st.subheader("Grounding Score per Prompt")
    grounding_data = pd.DataFrame({
        selected_label: {p["id"]: p.get("grounding_score", 0) for p in prompt_data},
        compare_label:  {p["id"]: p.get("grounding_score", 0) for p in prompt_data2},
    }).loc[common_ids]
    st.bar_chart(grounding_data)

    # Hallucination comparison
    st.subheader("Hallucination Rate per Prompt")
    halluc_data = pd.DataFrame({
        selected_label: {p["id"]: p.get("hallucination_rate", 0) for p in prompt_data},
        compare_label:  {p["id"]: p.get("hallucination_rate", 0) for p in prompt_data2},
    }).loc[common_ids]
    st.bar_chart(halluc_data)

    # Side-by-side per-prompt table
    st.subheader("Per-Prompt Metrics — Side by Side")
    merged = df.set_index("ID")[numeric_cols].add_suffix(f" [{selected_label}]").join(
        df2.set_index("ID")[numeric_cols].add_suffix(f" [{compare_label}]")
    )
    st.dataframe(merged.round(3), use_container_width=True)

    # Error taxonomy comparison
    st.subheader("Error Taxonomy Comparison")
    def get_taxonomy(pd_list):
        labels = [c["classification"] for p in pd_list for c in p.get("claims", [])]
        return pd.Series(labels).value_counts()

    tax1 = get_taxonomy(prompt_data).rename(selected_label)
    tax2 = get_taxonomy(prompt_data2).rename(compare_label)
    tax_df = pd.concat([tax1, tax2], axis=1).fillna(0).astype(int)
    st.bar_chart(tax_df)

    st.stop()  # Don't show single-run view when in compare mode

# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN VIEW
# ═══════════════════════════════════════════════════════════════════════════════
st.header(f"Overall Model Summary — `{selected_label}`")
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
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("FRANQ Score", selected.get("prompt_score", "—"))
    m2.metric("Hallucination Rate", selected.get("hallucination_rate", "—"))
    m3.metric("Grounding Score", selected.get("grounding_score", "—"))
    m4.metric("Consistency", selected.get("consistency_score", "—"))
    m5.metric("Calibration", selected.get("calibration", "—"))
