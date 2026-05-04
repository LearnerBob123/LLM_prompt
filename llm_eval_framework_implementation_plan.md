# Prompt-wise LLM Evaluation Framework for Fixed-Knowledge RAG Systems
## Complete Implementation Plan — Build From Scratch

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack & Prerequisites](#2-tech-stack--prerequisites)
3. [Repository Structure](#3-repository-structure)
4. [Phase 1 — Environment Setup](#phase-1--environment-setup)
5. [Phase 2 — Knowledge Base & Dataset Construction](#phase-2--knowledge-base--dataset-construction)
6. [Phase 3 — Generation Engine (RAG Pipeline)](#phase-3--generation-engine-rag-pipeline)
7. [Phase 4 — Claim Extraction Module](#phase-4--claim-extraction-module)
8. [Phase 5 — Metric Computation (Per-Claim)](#phase-5--metric-computation-per-claim)
9. [Phase 6 — Prompt-Level Aggregated Metrics](#phase-6--prompt-level-aggregated-metrics)
10. [Phase 7 — Final Scoring & Error Taxonomy](#phase-7--final-scoring--error-taxonomy)
11. [Phase 8 — Dashboard & Output UI](#phase-8--dashboard--output-ui)
12. [Phase 9 — End-to-End Pipeline Runner](#phase-9--end-to-end-pipeline-runner)
13. [MVP Cutoff Guide](#mvp-cutoff-guide)
14. [Testing Checklist](#testing-checklist)

---

## 1. Project Overview

**What you are building:**

> A Prompt-wise LLM Evaluation Framework that, given a fixed knowledge base and multiple prompts, evaluates how well a language model reasons, grounds its answers, and avoids hallucination — at the atomic claim level.

**Core insight:** Instead of scoring whole responses, you decompose each response into atomic factual claims and score every claim individually. This unlocks metrics that are impossible at the response level.

**Key differentiators over a basic RAG eval:**
- Claim-level decomposition (not response-level)
- Faithfulness vs. Factuality separation
- Uncertainty quantification (log-probability confidence)
- Prompt-wise analytics (not just dataset-level averages)
- Error taxonomy per claim (4-quadrant classification)

---

## 2. Tech Stack & Prerequisites

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| LLM API | On device HuggingFace (Mistral / Llama) |
| NLI Model | `cross-encoder/nli-deberta-v3-base` (HuggingFace) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (local) |
| Dashboard | Streamlit |
| Data Format | JSON |
| Env Management | `python-dotenv` |

**Install all dependencies:**

```bash
pip install openai sentence-transformers faiss-cpu transformers \
            torch streamlit pandas numpy scikit-learn \
            python-dotenv tqdm
```

Set up your `.env` file:

```
we use local api so no openai key but if we require we can use gemini api key
```

---

## 3. Repository Structure

Create this folder layout before writing any code:

```
llm_eval_framework/
│
├── data/
│   ├── knowledge_base/          # Your fixed documents (.txt files)
│   ├── prompts.json             # 50–100 evaluation prompts
│   └── ground_truth.json        # Optional reference answers
│
├── outputs/
│   ├── responses.json           # Raw LLM responses
│   ├── claims.json              # Extracted claims per response
│   └── results.json             # Final metric scores
│
├── modules/
│   ├── retriever.py             # FAISS-based document retrieval
│   ├── generator.py             # LLM answer generation
│   ├── claim_extractor.py       # Claim decomposition via LLM
│   ├── faithfulness.py          # NLI-based faithfulness scoring
│   ├── confidence.py            # Log-probability confidence
│   ├── parametric.py            # Context-free factuality scoring
│   ├── consistency.py           # Multi-sample semantic consistency
│   ├── contradiction.py         # NLI contradiction detection
│   ├── prompt_metrics.py        # All prompt-level aggregated scores
│   └── scorer.py                # Final FRANQ-style score combiner
│
├── pipeline.py                  # End-to-end runner
├── dashboard.py                 # Streamlit UI
├── config.py                    # Central config (model names, thresholds)
└── .env
```

Run this to create it instantly:

```bash
mkdir -p llm_eval_framework/{data/knowledge_base,outputs,modules}
touch llm_eval_framework/{pipeline.py,dashboard.py,config.py,.env}
touch llm_eval_framework/modules/{__init__,retriever,generator,claim_extractor,faithfulness,confidence,parametric,consistency,contradiction,prompt_metrics,scorer}.py
```

---

## Phase 1 — Environment Setup

### Step 1.1 — `config.py`

This file centralizes all settings. Edit only this file to change models or thresholds.

```python
# config.py
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")#as mentioned we avoid openai model rather try to use mistral models or gemini model as we have that but thef free limit only PREFERABLY WE USE THE OLLAMA ON DEVICE LOCAL MODELS
GENERATOR_MODEL = "gpt-4o"           # Model used for generation
CLAIM_EXTRACTOR_MODEL = "gpt-4o"     # Model used for claim decomposition
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

TOP_K_DOCS = 3                       # Retrieve top-3 context docs
NUM_CONSISTENCY_SAMPLES = 3          # Samples for consistency scoring
FAITHFULNESS_THRESHOLD = 0.5         # Below this → unfaithful
CONFIDENCE_THRESHOLD = 0.4           # Below this → low confidence

KNOWLEDGE_BASE_DIR = "data/knowledge_base/"
PROMPTS_FILE = "data/prompts.json"
GROUND_TRUTH_FILE = "data/ground_truth.json"
RESPONSES_FILE = "outputs/responses.json"
CLAIMS_FILE = "outputs/claims.json"
RESULTS_FILE = "outputs/results.json"
```

---

## Phase 2 — Knowledge Base & Dataset Construction

### Step 2.1 — Populate the Knowledge Base

Place 5–10 `.txt` files inside `data/knowledge_base/`. These are your fixed documents. They must remain **unchanged** for every prompt. Example topics: a Wikipedia article set, a product manual, a research paper collection.

Each file should be plain text, roughly 200–500 words. Example:

```
data/knowledge_base/
    doc1_climate_change.txt
    doc2_renewable_energy.txt
    doc3_carbon_capture.txt
    ...
```

### Step 2.2 — Create `data/prompts.json`

Write 50–100 prompts across three categories. Aim for roughly:
- 40% Factual QA
- 30% Reasoning QA
- 30% Ambiguous / edge-case queries

```json
[
  {
    "id": "p001",
    "prompt": "What is the main cause of rising global temperatures?",
    "type": "factual"
  },
  {
    "id": "p002",
    "prompt": "Why might renewable energy alone not solve climate change?",
    "type": "reasoning"
  },
  {
    "id": "p003",
    "prompt": "Is carbon capture a proven solution?",
    "type": "ambiguous"
  }
]
```

### Step 2.3 — Create `data/ground_truth.json` (Optional but recommended)

```json
{
  "p001": "The main cause of rising global temperatures is the increased concentration of greenhouse gases, primarily CO2 from fossil fuel combustion.",
  "p002": "Renewable energy addresses electricity generation but not all sectors...",
  "p003": "Carbon capture technologies exist but are not yet deployed at scale..."
}
```

### Step 2.4 — Build the FAISS Retriever (`modules/retriever.py`)

```python
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import KNOWLEDGE_BASE_DIR, EMBEDDING_MODEL, TOP_K_DOCS

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.documents = []
        self.index = None
        self._build_index()

    def _build_index(self):
        for fname in os.listdir(KNOWLEDGE_BASE_DIR):
            if fname.endswith(".txt"):
                with open(os.path.join(KNOWLEDGE_BASE_DIR, fname)) as f:
                    self.documents.append(f.read().strip())

        embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = TOP_K_DOCS) -> list[str]:
        query_vec = self.model.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(query_vec, top_k)
        return [self.documents[i] for i in indices[0]]
```

---

## Phase 3 — Generation Engine (RAG Pipeline)

### Step 3.1 — `modules/generator.py`

This module calls the LLM with and without context. The `without_context` flag is used later for parametric scoring.

```python
from openai import OpenAI
from config import OPENAI_API_KEY, GENERATOR_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(prompt: str, context_docs: list[str], with_context: bool = True) -> dict:
    context_str = "\n\n".join(context_docs) if with_context else ""

    system_msg = "You are a helpful assistant that answers questions based on the provided context."
    user_msg = f"Context:\n{context_str}\n\nQuestion: {prompt}" if with_context else f"Question: {prompt}"

    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        logprobs=True,          # REQUIRED for confidence scoring
        top_logprobs=1,
        temperature=0.0         # Deterministic for primary generation
    )

    choice = response.choices[0]
    token_logprobs = [t.logprob for t in choice.logprobs.content]

    return {
        "response": choice.message.content,
        "token_logprobs": token_logprobs
    }
```

> **Note:** `logprobs=True` is available in OpenAI's API for GPT-4o. If using HuggingFace, extract log-probs from model outputs manually.

---

## Phase 4 — Claim Extraction Module

This is the most critical module. It upgrades the project from a basic eval into a research-grade system.

### Step 4.1 — `modules/claim_extractor.py`

```python
from openai import OpenAI
from config import OPENAI_API_KEY, CLAIM_EXTRACTOR_MODEL
import json

client = OpenAI(api_key=OPENAI_API_KEY)

CLAIM_EXTRACTION_PROMPT = """
You are an expert fact-checker. Given a text response, extract all atomic factual claims.

Rules:
- Each claim must be a single, self-contained factual statement.
- Do NOT include opinions, hedges, or meta-commentary.
- Do NOT merge two facts into one claim.
- Output ONLY a JSON list of strings. No preamble. No explanation.

Example input: "The Eiffel Tower was built in 1889 and is located in Paris. It is 330 meters tall."
Example output: ["The Eiffel Tower was built in 1889.", "The Eiffel Tower is located in Paris.", "The Eiffel Tower is 330 meters tall."]

Now extract claims from this response:
{response}
"""

def extract_claims(response: str) -> list[str]:
    prompt = CLAIM_EXTRACTION_PROMPT.format(response=response)
    result = client.chat.completions.create(
        model=CLAIM_EXTRACTOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    raw = result.choices[0].message.content.strip()
    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        claims = json.loads(raw)
        return [c for c in claims if isinstance(c, str)]
    except json.JSONDecodeError:
        return []
```

---

## Phase 5 — Metric Computation (Per-Claim)

Each of the following modules scores a single claim `c_i` against the context or model behavior.

### Step 5.1 — Faithfulness via NLI (`modules/faithfulness.py`)

```python
from transformers import pipeline
from config import NLI_MODEL

_nli = None

def _get_nli():
    global _nli
    if _nli is None:
        _nli = pipeline("text-classification", model=NLI_MODEL)
    return _nli

def faithfulness_score(claim: str, context_docs: list[str]) -> float:
    """Returns P(context entails claim). Range: [0, 1]."""
    nli = _get_nli()
    context = " ".join(context_docs)
    # Premise = context, Hypothesis = claim
    input_text = f"{context} [SEP] {claim}"
    result = nli(input_text, truncation=True, max_length=512)[0]

    label_map = {"ENTAILMENT": 1.0, "NEUTRAL": 0.5, "CONTRADICTION": 0.0}
    # DeBERTa NLI returns labels as entailment/neutral/contradiction
    label = result["label"].upper()
    # Weighted by score for smoother output
    base = label_map.get(label, 0.5)
    return round(base * result["score"] + (1 - result["score"]) * 0.5, 4)
```

### Step 5.2 — Contradiction Score (`modules/contradiction.py`)

```python
from transformers import pipeline
from config import NLI_MODEL

_nli = None

def _get_nli():
    global _nli
    if _nli is None:
        _nli = pipeline("text-classification", model=NLI_MODEL)
    return _nli

def contradiction_score(claim: str, context_docs: list[str]) -> float:
    """Returns probability that the claim contradicts the context. Range: [0, 1]."""
    nli = _get_nli()
    context = " ".join(context_docs)
    input_text = f"{context} [SEP] {claim}"
    result = nli(input_text, truncation=True, max_length=512)[0]
    label = result["label"].upper()
    if label == "CONTRADICTION":
        return round(result["score"], 4)
    return 0.0
```

### Step 5.3 — Confidence via Log-Probability (`modules/confidence.py`)

```python
import math

def confidence_score(token_logprobs: list[float]) -> float:
    """
    Computes normalized log-likelihood (inverse perplexity).
    Formula: exp( avg(log P(token_t)) )
    Range: (0, 1] — higher is more confident.
    """
    if not token_logprobs:
        return 0.0
    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    return round(math.exp(avg_logprob), 4)
```

> **Note:** `token_logprobs` come from the `generate_answer()` call. This gives a whole-response confidence. For claim-level confidence, you can re-query the model with just the claim as output.

### Step 5.4 — Parametric Knowledge Score (`modules/parametric.py`)

```python
from modules.generator import generate_answer

def parametric_score(claim: str, prompt: str, nli_pipeline) -> float:
    """
    Re-generates response WITHOUT context, then checks if the claim is still supported.
    This measures model's internal (parametric) knowledge of the claim.
    """
    result = generate_answer(prompt, context_docs=[], with_context=False)
    context_free_response = result["response"]

    # Check NLI: does the context-free response entail the claim?
    input_text = f"{context_free_response} [SEP] {claim}"
    nli_result = nli_pipeline(input_text, truncation=True, max_length=512)[0]
    label = nli_result["label"].upper()
    label_map = {"ENTAILMENT": 1.0, "NEUTRAL": 0.5, "CONTRADICTION": 0.0}
    base = label_map.get(label, 0.5)
    return round(base * nli_result["score"] + (1 - nli_result["score"]) * 0.5, 4)
```

### Step 5.5 — Semantic Consistency (`modules/consistency.py`)

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from modules.generator import generate_answer
from config import EMBEDDING_MODEL, NUM_CONSISTENCY_SAMPLES

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def consistency_score(prompt: str, context_docs: list[str]) -> float:
    """
    Generates k responses, embeds them, and computes average pairwise cosine similarity.
    Range: [0, 1] — higher means more consistent outputs.
    """
    samples = []
    for _ in range(NUM_CONSISTENCY_SAMPLES):
        result = generate_answer(prompt, context_docs)
        samples.append(result["response"])

    embedder = _get_embedder()
    embeddings = embedder.encode(samples)
    n = len(embeddings)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            sims.append(sim)
    return round(float(np.mean(sims)), 4) if sims else 1.0
```

---

## Phase 6 — Prompt-Level Aggregated Metrics

### Step 6.1 — `modules/prompt_metrics.py`

All the per-prompt metrics are computed here from the list of per-claim scores.

```python
import numpy as np
from config import FAITHFULNESS_THRESHOLD, CONFIDENCE_THRESHOLD

def grounding_score(faithfulness_scores: list[float]) -> float:
    """Average faithfulness across all claims."""
    return round(float(np.mean(faithfulness_scores)), 4) if faithfulness_scores else 0.0

def hallucination_rate(faithfulness_scores: list[float], confidence_scores: list[float]) -> float:
    """
    % of claims with both low faithfulness AND low confidence.
    These are likely hallucinated claims.
    """
    if not faithfulness_scores:
        return 0.0
    hallucinated = sum(
        1 for f, c in zip(faithfulness_scores, confidence_scores)
        if f < FAITHFULNESS_THRESHOLD and c < CONFIDENCE_THRESHOLD
    )
    return round(hallucinated / len(faithfulness_scores), 4)

def confidence_calibration(confidence_scores: list[float], faithfulness_scores: list[float]) -> float:
    """
    Pearson correlation between confidence and faithfulness.
    High correlation = well-calibrated model.
    Range: [-1, 1]
    """
    if len(confidence_scores) < 2:
        return 0.0
    return round(float(np.corrcoef(confidence_scores, faithfulness_scores)[0, 1]), 4)

def relevance_score(prompt: str, response: str, embedder) -> float:
    """Cosine similarity between prompt and response embeddings."""
    vecs = embedder.encode([prompt, response])
    from sklearn.metrics.pairwise import cosine_similarity
    return round(float(cosine_similarity([vecs[0]], [vecs[1]])[0][0]), 4)

def completeness_score(response: str, ground_truth: str, embedder) -> float:
    """Semantic similarity of response to ground truth. Requires ground truth."""
    if not ground_truth:
        return None
    vecs = embedder.encode([response, ground_truth])
    from sklearn.metrics.pairwise import cosine_similarity
    return round(float(cosine_similarity([vecs[0]], [vecs[1]])[0][0]), 4)

def robustness_score(prompt: str, context_docs: list[str], generate_fn, embedder) -> float:
    """
    Paraphrase the prompt → re-generate → compare outputs.
    Measures output stability to prompt wording variation.
    """
    paraphrase_prompt = f"Paraphrase this question in a different way: {prompt}"
    from openai import OpenAI
    from config import OPENAI_API_KEY, GENERATOR_MODEL
    client = OpenAI(api_key=OPENAI_API_KEY)
    paraphrase_result = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": paraphrase_prompt}],
        temperature=0.0
    )
    paraphrased = paraphrase_result.choices[0].message.content.strip()
    original_result = generate_fn(prompt, context_docs)
    paraphrased_result = generate_fn(paraphrased, context_docs)
    vecs = embedder.encode([original_result["response"], paraphrased_result["response"]])
    from sklearn.metrics.pairwise import cosine_similarity
    return round(float(cosine_similarity([vecs[0]], [vecs[1]])[0][0]), 4)
```

---

## Phase 7 — Final Scoring & Error Taxonomy

### Step 7.1 — `modules/scorer.py`

```python
from config import FAITHFULNESS_THRESHOLD, CONFIDENCE_THRESHOLD

def franq_score(faithfulness: float, confidence: float, parametric: float,
                contradiction: float, consistency: float) -> float:
    """
    FRANQ-style composite claim score.
    When faithfulness is high → reward grounded answer.
    When faithfulness is low → fall back to parametric knowledge.
    Penalize contradictions and reward consistency.
    """
    base = faithfulness * confidence + (1 - faithfulness) * parametric
    base *= (1 - contradiction)
    base *= consistency
    return round(base, 4)

def classify_claim(faithfulness: float, parametric: float) -> str:
    """
    Error taxonomy (4-quadrant):
      Faithful + True  → ✅ Grounded Correct
      Faithful + False → ⚠️ Grounded Incorrect (context error)
      Unfaithful + True → 🤔 Lucky Correct (hallucinated but right)
      Unfaithful + False → ❌ Hallucination
    """
    is_faithful = faithfulness >= FAITHFULNESS_THRESHOLD
    is_factual = parametric >= CONFIDENCE_THRESHOLD

    if is_faithful and is_factual:
        return "✅ Grounded Correct"
    elif is_faithful and not is_factual:
        return "⚠️ Grounded Incorrect"
    elif not is_faithful and is_factual:
        return "🤔 Lucky Correct"
    else:
        return "❌ Hallucination"

def prompt_score(claim_scores: list[float]) -> float:
    """Average FRANQ score across all claims in a prompt."""
    return round(sum(claim_scores) / len(claim_scores), 4) if claim_scores else 0.0

def model_score(prompt_scores: list[float]) -> float:
    """Average prompt score across all prompts = overall model score."""
    return round(sum(prompt_scores) / len(prompt_scores), 4) if prompt_scores else 0.0
```

---

## Phase 8 — Dashboard & Output UI

### Step 8.1 — `dashboard.py` (Streamlit)

```python
import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="LLM Eval Dashboard", layout="wide")
st.title("📊 Prompt-wise LLM Evaluation Framework")

@st.cache_data
def load_results():
    with open("outputs/results.json") as f:
        return json.load(f)

results = load_results()

# ── Summary Metrics ──────────────────────────────────────────────
st.header("🔢 Overall Model Summary")
prompt_data = results["prompts"]
df = pd.DataFrame([{
    "Prompt ID": p["id"],
    "Type": p["type"],
    "Prompt Score": p["prompt_score"],
    "Grounding": p["grounding_score"],
    "Hallucination Rate": p["hallucination_rate"],
    "Consistency": p["consistency_score"],
    "Calibration": p["calibration"],
    "Relevance": p["relevance_score"],
    "Robustness": p["robustness_score"],
} for p in prompt_data])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model Score", round(results["model_score"], 3))
col2.metric("Avg Hallucination Rate", round(df["Hallucination Rate"].mean(), 3))
col3.metric("Avg Grounding", round(df["Grounding"].mean(), 3))
col4.metric("Avg Consistency", round(df["Consistency"].mean(), 3))

# ── Per-Prompt Table ──────────────────────────────────────────────
st.header("📋 Per-Prompt Metrics")
st.dataframe(df, use_container_width=True)

# ── Drill-Down ────────────────────────────────────────────────────
st.header("🔍 Prompt Drill-Down")
selected_id = st.selectbox("Select a Prompt", [p["id"] for p in prompt_data])
selected = next(p for p in prompt_data if p["id"] == selected_id)

st.subheader("Prompt")
st.write(selected["prompt"])
st.subheader("Response")
st.write(selected["response"])

st.subheader("Extracted Claims")
claim_rows = []
for c in selected["claims"]:
    claim_rows.append({
        "Claim": c["text"],
        "Faithfulness": c["faithfulness"],
        "Confidence": c["confidence"],
        "Parametric": c["parametric"],
        "Contradiction": c["contradiction"],
        "FRANQ Score": c["franq_score"],
        "Classification": c["classification"]
    })

st.dataframe(pd.DataFrame(claim_rows), use_container_width=True)

# ── Error Taxonomy Chart ──────────────────────────────────────────
st.subheader("Error Taxonomy Distribution")
all_classifications = [c["classification"] for p in prompt_data for c in p["claims"]]
taxonomy_df = pd.Series(all_classifications).value_counts().reset_index()
taxonomy_df.columns = ["Category", "Count"]
st.bar_chart(taxonomy_df.set_index("Category"))
```

Run the dashboard with:

```bash
streamlit run dashboard.py
```

---

## Phase 9 — End-to-End Pipeline Runner

### Step 9.1 — `pipeline.py`

This is the single file you run to execute everything from start to finish.

```python
import json
from tqdm import tqdm
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer

from config import *
from modules.retriever import Retriever
from modules.generator import generate_answer
from modules.claim_extractor import extract_claims
from modules.faithfulness import faithfulness_score
from modules.contradiction import contradiction_score
from modules.confidence import confidence_score
from modules.parametric import parametric_score
from modules.consistency import consistency_score
from modules.scorer import franq_score, classify_claim, prompt_score, model_score
from modules.prompt_metrics import (
    grounding_score, hallucination_rate, confidence_calibration,
    relevance_score, completeness_score, robustness_score
)

def run():
    print("Loading models...")
    retriever = Retriever()
    nli = hf_pipeline("text-classification", model=NLI_MODEL)
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)

    try:
        with open(GROUND_TRUTH_FILE) as f:
            ground_truths = json.load(f)
    except FileNotFoundError:
        ground_truths = {}

    all_prompt_results = []
    all_prompt_scores = []

    for item in tqdm(prompts, desc="Evaluating prompts"):
        pid = item["id"]
        prompt = item["prompt"]
        ptype = item.get("type", "factual")

        # Step 1: Retrieve context
        context_docs = retriever.retrieve(prompt)

        # Step 2: Generate response (with context)
        gen = generate_answer(prompt, context_docs, with_context=True)
        response = gen["response"]
        token_logprobs = gen["token_logprobs"]

        # Step 3: Extract claims
        claims = extract_claims(response)

        # Step 4: Score each claim
        conf = confidence_score(token_logprobs)   # Response-level (proxy for all claims)
        consist = consistency_score(prompt, context_docs)

        claim_results = []
        faith_scores, conf_scores = [], []

        for claim_text in claims:
            faith = faithfulness_score(claim_text, context_docs)
            contra = contradiction_score(claim_text, context_docs)
            param = parametric_score(claim_text, prompt, nli)
            score = franq_score(faith, conf, param, contra, consist)
            label = classify_claim(faith, param)

            faith_scores.append(faith)
            conf_scores.append(conf)

            claim_results.append({
                "text": claim_text,
                "faithfulness": faith,
                "confidence": conf,
                "parametric": param,
                "contradiction": contra,
                "consistency": consist,
                "franq_score": score,
                "classification": label
            })

        # Step 5: Prompt-level metrics
        p_score = prompt_score([c["franq_score"] for c in claim_results])
        gt = ground_truths.get(pid, "")

        prompt_result = {
            "id": pid,
            "type": ptype,
            "prompt": prompt,
            "response": response,
            "context": context_docs,
            "claims": claim_results,
            "prompt_score": p_score,
            "grounding_score": grounding_score(faith_scores),
            "hallucination_rate": hallucination_rate(faith_scores, conf_scores),
            "calibration": confidence_calibration(conf_scores, faith_scores),
            "consistency_score": consist,
            "relevance_score": relevance_score(prompt, response, embedder),
            "completeness_score": completeness_score(response, gt, embedder),
            "robustness_score": robustness_score(prompt, context_docs, generate_answer, embedder)
        }

        all_prompt_results.append(prompt_result)
        all_prompt_scores.append(p_score)

    final = {
        "model_score": model_score(all_prompt_scores),
        "prompts": all_prompt_results
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nDone. Results saved to {RESULTS_FILE}")
    print(f"Overall Model Score: {final['model_score']}")

if __name__ == "__main__":
    run()
```

---

## MVP Cutoff Guide

Use this if you're short on time. Build in this exact priority order:

| Priority | Module | Time Estimate |
|---|---|---|
| ✅ MUST | FAISS Retriever | 30 min |
| ✅ MUST | LLM Generator (with logprobs) | 30 min |
| ✅ MUST | Claim Extractor | 45 min |
| ✅ MUST | Faithfulness (NLI) | 45 min |
| ✅ MUST | Confidence (log-prob) | 20 min |
| ✅ MUST | FRANQ Score + Classifier | 30 min |
| ✅ MUST | pipeline.py (connecting all of the above) | 45 min |
| 🟡 NICE | Parametric Score | 30 min |
| 🟡 NICE | Contradiction Score | 20 min |
| 🟡 NICE | Consistency Score | 30 min |
| 🟡 NICE | Robustness Score | 30 min |
| 🟢 EXTRA | Streamlit Dashboard | 1 hour |

**Minimum viable demo output per prompt:**
1. The prompt text
2. The response
3. All extracted claims as a table
4. Per-claim: Faithfulness, Confidence, FRANQ Score, Classification label
5. Summary: Hallucination Rate, Grounding Score

---

## Testing Checklist

Before your demo, verify each of these:

- [ ] `Retriever.retrieve()` returns 3 relevant documents for a test query
- [ ] `generate_answer()` returns both `response` and non-empty `token_logprobs`
- [ ] `extract_claims()` returns a Python list of strings (not raw JSON text)
- [ ] `faithfulness_score()` returns a float in `[0, 1]`
- [ ] `contradiction_score()` returns 0.0 for clearly non-contradicting claims
- [ ] `confidence_score()` returns a value between 0 and 1
- [ ] `franq_score()` is lower for claims marked ❌ Hallucination
- [ ] `classify_claim()` correctly labels obviously faithful claims as ✅
- [ ] `pipeline.py` writes a valid `results.json` file after running
- [ ] `dashboard.py` loads without errors and displays the results table
- [ ] Hallucination Rate is higher for ambiguous prompt types than factual ones

---

*End of Implementation Plan*
