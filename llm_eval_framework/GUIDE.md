# LLM Evaluation Framework — Complete Developer Guide

> **Purpose:** This document explains every file, every setting, every function, and how to modify the framework to suit your needs.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [How to Run](#3-how-to-run)
4. [config.py — All Settings Explained](#4-configpy--all-settings-explained)
5. [Data Files](#5-data-files)
6. [Modules — Deep Dive](#6-modules--deep-dive)
   - [retriever.py](#61-retrieverpy)
   - [generator.py](#62-generatorpy)
   - [claim_extractor.py](#63-claim_extractorpy)
   - [faithfulness.py](#64-faithfulnesspy)
   - [contradiction.py](#65-contradictionpy)
   - [confidence.py](#66-confidencepy)
   - [parametric.py](#67-parametricpy)
   - [consistency.py](#68-consistencypy)
   - [prompt_metrics.py](#69-prompt_metricspy)
   - [scorer.py](#610-scorerpy)
7. [pipeline.py — End-to-End Runner](#7-pipelinepy--end-to-end-runner)
8. [dashboard.py — Streamlit UI](#8-dashboardpy--streamlit-ui)
9. [Output Format — results.json](#9-output-format--resultsjson)
10. [How to Modify — Common Scenarios](#10-how-to-modify--common-scenarios)
11. [Models Reference](#11-models-reference)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Overview

This framework evaluates how well a language model answers questions when given a fixed knowledge base. Instead of scoring whole responses, it:

1. **Retrieves** relevant documents from the knowledge base using FAISS vector search
2. **Generates** an LLM answer using those documents as context (RAG)
3. **Extracts** atomic factual claims from the answer (e.g. splits "Paris is the capital and has 2M people" into two separate claims)
4. **Scores** every claim individually on faithfulness, confidence, contradiction, parametric knowledge
5. **Aggregates** claim scores into prompt-level metrics (grounding, hallucination rate, calibration, etc.)
6. **Classifies** each claim into one of 4 error taxonomy categories
7. **Displays** everything in a Streamlit dashboard

**Why claim-level?** Response-level scoring misses which specific statements are hallucinated. Claim-level scoring tells you exactly which sentence in a 10-sentence answer is wrong and why.

---

## 2. Folder Structure

```
llm_eval_framework/
│
├── config.py                    ← ALL settings live here — edit this file first
├── pipeline.py                  ← Main runner script
├── dashboard.py                 ← Streamlit UI
├── .env                         ← Environment variables (Ollama URL, API keys)
├── GUIDE.md                     ← This file
│
├── data/
│   ├── knowledge_base/          ← Your .txt documents (fixed, never modified)
│   │   ├── doc01_history_of_computers.txt
│   │   ├── doc02_internet_and_networking.txt
│   │   ├── ... (10 files total)
│   ├── prompts.json             ← 20 evaluation prompts with IDs and types
│   └── ground_truth.json        ← Reference answers for factual prompts (optional)
│
├── outputs/
│   └── results.json             ← Written by pipeline.py, read by dashboard.py
│
└── modules/
    ├── __init__.py
    ├── retriever.py             ← FAISS document retrieval
    ├── generator.py             ← Ollama LLM answer generation
    ├── claim_extractor.py       ← Decomposes response into atomic claims
    ├── faithfulness.py          ← NLI: does context support the claim?
    ├── contradiction.py         ← NLI: does context contradict the claim?
    ├── confidence.py            ← Log-probability confidence score
    ├── parametric.py            ← LLM internal knowledge check
    ├── consistency.py           ← Multi-sample semantic consistency
    ├── prompt_metrics.py        ← Prompt-level aggregated metrics
    └── scorer.py                ← FRANQ composite score + error taxonomy
```

---

## 3. How to Run

### Prerequisites

**1. Install Ollama** (if not done already)
- Download from: https://ollama.com/download/windows
- After install, open a terminal and pull the models:

```powershell
ollama pull mistral       # ~4.7 GB — primary answer generator
ollama pull qwen2.5       # ~4.7 GB — claim extractor (best JSON output)
ollama pull gemma3:4b     # ~3.3 GB — used for consistency sampling
```

**2. Verify Ollama is running**
```powershell
ollama list               # should show the 3 models
```

### Run the pipeline

All commands are run from inside `llm_eval_framework/` using the `gpu` conda environment.

```powershell
# Navigate to the framework folder
Set-Location "C:\Users\91930\Desktop\GENAI Final Project\llm_eval_framework"

# Shortcut — set the Python path
$py = "C:\Users\91930\anaconda3\envs\gpu\python.exe"

# ── Mini run — 5 prompts, fast mode (best for testing) ──
& $py pipeline.py --mini

# ── Full run — 20 prompts, fast mode (default) ──
& $py pipeline.py

# ── Full run with all metrics enabled ──
& $py pipeline.py --full

# ── Launch dashboard ──
& "C:\Users\91930\anaconda3\envs\gpu\Scripts\streamlit.exe" run dashboard.py
```

### What `--mini` vs `--full` means

| Flag | Prompts | FAST_MODE | Extra LLM calls | Approx. time |
|---|---|---|---|---|
| `--mini` | 5 | True | None | ~1–2 min |
| _(none)_ | 20 | True (from config) | None | ~5–8 min |
| `--full` | 20 | False | parametric + consistency + robustness | ~20–40 min |
| `--fast` | 20 | True (forced) | None | ~5–8 min |

> **First run** also downloads the NLI model (`cross-encoder/nli-deberta-v3-base`, ~500 MB) from HuggingFace automatically.

---

## 4. `config.py` — All Settings Explained

This is the **only file you need to edit** to change models, thresholds, or paths. Everything else reads from here.

```python
# ── Ollama Settings ──────────────────────────────────────────────────────────

OLLAMA_HOST = "http://localhost:11434"
# Where Ollama is running. Change this if Ollama is on a different machine or port.

GENERATOR_MODEL = "mistral"
# The Ollama model used to generate RAG answers.
# Change to: "llama3", "gemma3:4b", "phi4", "deepseek-r1:7b", etc.
# Any model you have pulled with `ollama pull <name>` works here.

CLAIM_EXTRACTOR_MODEL = "qwen2.5"
# The Ollama model used to decompose responses into atomic claims.
# Needs good JSON output. qwen2.5 is recommended. You can also use "mistral".

CONSISTENCY_MODEL = "gemma3:4b"
# Used for the consistency score (runs 3 re-samples of the same prompt).
# Use a smaller/faster model here to save time.

# ── Local Model Settings ─────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Sentence-transformers model used by the FAISS retriever and for
# relevance/completeness/robustness cosine similarity comparisons.
# Other options: "all-mpnet-base-v2" (more accurate, slower),
#                "paraphrase-MiniLM-L6-v2" (faster, less accurate)

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
# HuggingFace CrossEncoder NLI model. Used by faithfulness.py and contradiction.py.
# Alternatives: "cross-encoder/nli-MiniLM2-L6-H768" (faster, slightly less accurate)
# Label order is fixed: index 0=contradiction, index 1=entailment, index 2=neutral

# ── Pipeline Settings ────────────────────────────────────────────────────────

TOP_K_DOCS = 3
# How many knowledge base documents to retrieve per prompt.
# Increase to 5 if you have many documents and want broader context.
# Decrease to 1 for very focused retrieval.

NUM_CONSISTENCY_SAMPLES = 3
# How many re-samples to generate for consistency scoring.
# Only used when FAST_MODE = False.

FAST_MODE = True
# True  → skip parametric, consistency, robustness scoring (faster, fewer LLM calls)
# False → compute all metrics (slower but more complete)
# Can be overridden per run with --fast or --full flags

# ── Thresholds ───────────────────────────────────────────────────────────────

FAITHFULNESS_THRESHOLD = 0.5
# Claims with faithfulness BELOW this are considered "unfaithful" (not grounded in context).
# Increase to 0.7 for stricter grounding requirements.
# Decrease to 0.3 if your NLI model tends to score low.

CONFIDENCE_THRESHOLD = 0.4
# Claims with confidence BELOW this are considered "low confidence".
# A claim that is both unfaithful AND low confidence is classified as ❌ Hallucination.

# ── Paths ────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE_DIR = "data/knowledge_base/"
PROMPTS_FILE = "data/prompts.json"
GROUND_TRUTH_FILE = "data/ground_truth.json"
RESULTS_FILE = "outputs/results.json"
```

---

## 5. Data Files

### `data/knowledge_base/*.txt`

These are the **fixed documents** the model is evaluated against. The key constraint: they never change during evaluation. The model is expected to answer using only these documents.

**Current documents (10 files, ~300 words each):**
- `doc01` — History of computers (ENIAC, transistors, microprocessors)
- `doc02` — Internet and networking (TCP/IP, DNS, HTTP, Wi-Fi)
- `doc03` — Artificial intelligence (ML, deep learning, Transformers, LLMs, RAG)
- `doc04` — World geography (continents, oceans, rivers, deserts)
- `doc05` — Climate and weather (greenhouse effect, global warming, seasons)
- `doc06` — Human biology (cells, circulatory system, brain, immune system)
- `doc07` — Space exploration (Sputnik, Apollo, ISS, Mars rovers, JWST)
- `doc08` — Economics basics (supply/demand, GDP, inflation, stock markets)
- `doc09` — World history (ancient civilisations, World Wars, Cold War)
- `doc10` — Mathematics (algebra, calculus, statistics, prime numbers, geometry)

**To add your own documents:**
1. Drop a `.txt` file into `data/knowledge_base/`
2. The retriever auto-discovers all `.txt` files — no code changes needed
3. Restart the pipeline; the FAISS index rebuilds automatically each run

**To replace with your own topic:**
1. Delete the existing `.txt` files
2. Add your own `.txt` files
3. Update `data/prompts.json` and `data/ground_truth.json` to match

---

### `data/prompts.json`

Format:
```json
[
  {
    "id": "p001",
    "prompt": "What was ENIAC and when was it completed?",
    "type": "factual"
  }
]
```

**Fields:**
- `id` — unique identifier, used to link to ground truth. Must be unique.
- `prompt` — the question asked to the model
- `type` — one of `"factual"`, `"reasoning"`, or `"ambiguous"`. Used in dashboard grouping.

**Current distribution:** 8 factual, 6 reasoning, 6 ambiguous (20 total)

**To add prompts:** Simply append more objects to the JSON array. No code changes needed.

---

### `data/ground_truth.json`

Format:
```json
{
  "p001": "Reference answer text...",
  "p006": "Reference answer text..."
}
```

Only prompts with entries here get a `completeness_score`. Others get `null`. It is optional — the pipeline runs fine without it. Currently only the 8 factual prompts have entries.

---

## 6. Modules — Deep Dive

### 6.1 `retriever.py`

**What it does:** Builds a FAISS index over all knowledge base documents and retrieves the top-K most relevant documents for a query.

**How it works:**
1. Loads all `.txt` files from `KNOWLEDGE_BASE_DIR`
2. Encodes them into vectors using `SentenceTransformer(EMBEDDING_MODEL)`
3. Stores vectors in a `faiss.IndexFlatL2` (exact L2 distance search)
4. On `retrieve(query)`: encodes the query, finds the `TOP_K_DOCS` nearest documents

**Key class and method:**
```python
retriever = Retriever()
docs = retriever.retrieve("What is DNS?")
# Returns: list of 3 document strings (most relevant first)
```

**How to modify:**
- **Change number of retrieved docs:** Edit `TOP_K_DOCS` in `config.py`
- **Use approximate search (faster for large KB):** Replace `IndexFlatL2` with `faiss.IndexIVFFlat` (requires training)
- **Add metadata to documents:** Extend the class to store `(filename, text)` tuples and return filenames alongside text

---

### 6.2 `generator.py`

**What it does:** Sends a prompt + context to an Ollama LLM and returns the response along with per-token log-probabilities.

**Key function:**
```python
result = generate_answer(prompt, context_docs, with_context=True)
# result["response"]        → the generated text
# result["token_logprobs"]  → list of floats (log P per token), used by confidence.py
```

**Parameters:**
- `prompt` — the question
- `context_docs` — list of document strings from the retriever
- `with_context=True` — if False, sends the question WITHOUT any context (used by `parametric.py`)
- `model` — optional override; defaults to `GENERATOR_MODEL` from config

**Ollama-specific details:**
- Uses `ollama.Client(host=OLLAMA_HOST).chat(..., logprobs=True)`
- Logprobs returned as a list of `Logprob` objects with a `.logprob` float attribute
- `temperature=0.0` for deterministic primary generation

**How to modify:**
- **Switch to a different model for one run:** Pass `model="llama3"` as argument
- **Use Gemini instead of Ollama:** Replace the `ollama.Client().chat()` call with the `google-generativeai` SDK; map the response to the same return dict shape
- **Adjust generation length:** Add `"num_predict": 512` to the `options` dict
- **Use streaming output:** Set `stream=True` and iterate over chunks; logprobs are available in streaming too

---

### 6.3 `claim_extractor.py`

**What it does:** Takes a model response and breaks it into a list of atomic factual claims using an LLM.

**Key function:**
```python
claims = extract_claims("The Eiffel Tower was built in 1889 and is in Paris.")
# Returns: ["The Eiffel Tower was built in 1889.", "The Eiffel Tower is in Paris."]
```

**How it works:**
- Sends a carefully designed prompt to `CLAIM_EXTRACTOR_MODEL` (qwen2.5)
- Uses `format="json"` in Ollama to force structured JSON output
- Expects JSON with a `"claims"` key containing a list of strings
- Has multiple fallback parsers in case the model wraps output in code fences

**What counts as a "claim":**
- ✅ Single factual statement: *"Water boils at 100°C at sea level."*
- ❌ Opinion: *"This is an important fact."*
- ❌ Hedge: *"It might be the case that..."*
- ❌ Merged claims: *"Paris is the capital and has 2 million people."*

**How to modify:**
- **Change the extraction prompt:** Edit `CLAIM_EXTRACTION_PROMPT` in the file. The `{response}` placeholder is required.
- **Use a different model:** Change `CLAIM_EXTRACTOR_MODEL` in `config.py`
- **Limit the number of claims:** Add a slice `return claims[:10]` before the final return to cap at 10 claims per response
- **Filter out very short claims:** Add `if len(c) > 20` to the list comprehension

---

### 6.4 `faithfulness.py`

**What it does:** Measures how well the context documents support a claim using NLI (Natural Language Inference).

**Key function:**
```python
score = faithfulness_score("The Eiffel Tower was built in 1889.", context_docs)
# Returns: float in [0, 1]
# High score → context strongly entails (supports) the claim
# Low score  → context does not support the claim
```

**How it works:**
- Uses `CrossEncoder(NLI_MODEL)` from `sentence_transformers`
- Input pair: `(context_text, claim)` — context is the "premise", claim is the "hypothesis"
- Calls `predict(..., apply_softmax=True)` to get a probability distribution over 3 labels
- **Label index order for `cross-encoder/nli-deberta-v3-base`:**
  - Index 0 → Contradiction
  - Index 1 → Entailment ← this is returned as faithfulness_score
  - Index 2 → Neutral
- Context is truncated to 1500 characters to avoid token limits

**How to modify:**
- **Use a different NLI model:** Change `NLI_MODEL` in `config.py`. **Important:** verify label order for the new model — it may differ.
- **Use a weighted score instead of raw entailment:** E.g. `0.7 * entailment + 0.3 * (1 - contradiction)` for a more nuanced score
- **Score against individual documents** (instead of concatenated): Loop over `context_docs`, score each separately, take the max

---

### 6.5 `contradiction.py`

**What it does:** Measures the probability that a claim actively contradicts the context.

**Key function:**
```python
score = contradiction_score("The Eiffel Tower was built in 2005.", context_docs)
# Returns: float in [0, 1]
# High score → claim contradicts the context
# 0.0        → no contradiction detected
```

**How it works:**
- Uses the same NLI singleton as `faithfulness.py` (shared via `_get_nli()`)
- Same input format: `(context, claim)`
- Returns `scores[0][0]` — the contradiction probability (index 0)

**Relationship to faithfulness:**
- These are NOT exact inverses. A claim can have:
  - Low faithfulness + low contradiction = neutral (context says nothing about it)
  - Low faithfulness + high contradiction = clear hallucination
  - High faithfulness + low contradiction = well-grounded claim

---

### 6.6 `confidence.py`

**What it does:** Converts per-token log-probabilities into a single confidence score for the entire response.

**Key function:**
```python
score = confidence_score([-0.3, -0.5, -1.2, -0.8])
# Returns: float in (0, 1]
# Higher → model generated the response with high average token probability
# Lower  → model was uncertain about many tokens
```

**Formula:**
$$\text{confidence} = \exp\left(\frac{1}{N} \sum_{t=1}^{N} \log P(\text{token}_t)\right)$$

This is equivalent to the geometric mean of per-token probabilities, or the inverse of perplexity raised to the power of 1/N.

**Important notes:**
- Returns `0.0` when no logprobs are available (some Ollama model variants don't return them)
- Clamps input to range `[-20, 0]` to prevent math overflow
- This is a **response-level** score, used as a proxy for all claims from that response. True claim-level confidence would require re-querying the model for each claim — computationally expensive.

**How to modify:**
- **Use a threshold-based binary score:** `return 1.0 if score >= CONFIDENCE_THRESHOLD else 0.0`
- **Smooth with Bayesian shrinkage:** `return (score * len(logprobs) + 0.5) / (len(logprobs) + 1)` to avoid extreme values with few tokens

---

### 6.7 `parametric.py`

**What it does:** Measures whether the model's internal (parametric) knowledge supports a claim, independent of the retrieved context.

**Key function:**
```python
score = parametric_score(claim_text, original_prompt, nli_pipeline)
# Returns: float in [0, 1]
# High score → model knows this fact from training data (not just from context)
# Low score  → model wouldn't say this without the context
```

**How it works:**
1. Re-generates an answer for the same prompt **without** any context (`with_context=False`)
2. Runs NLI on `(context_free_response, claim)` to check if the response entails the claim

**FAST_MODE=True:** Returns `0.5` (neutral) immediately. This prevents extra LLM calls. When FAST_MODE is True, the parametric score does not influence the FRANQ score meaningfully.

**How to modify:**
- **Always enable, ignore FAST_MODE:** Delete the `if FAST_MODE: return 0.5` block
- **Cache context-free responses:** The current code re-generates for every claim in a response. For efficiency, generate once per prompt and cache it.

---

### 6.8 `consistency.py`

**What it does:** Measures how semantically similar the model's answers are when asked the same question multiple times at higher temperature.

**Key function:**
```python
score = consistency_score(prompt, context_docs)
# Returns: float in [0, 1]
# 1.0 → all re-samples say the same thing (highly consistent)
# 0.0 → completely different answers each time
```

**How it works:**
1. Generates `NUM_CONSISTENCY_SAMPLES` (default 3) responses using `CONSISTENCY_MODEL` at `temperature=0.7`
2. Encodes all responses using `SentenceTransformer`
3. Computes pairwise cosine similarity between all embeddings
4. Returns the mean of all pairwise similarities

**FAST_MODE=True:** Returns `1.0` immediately (assumes perfect consistency). Since all claims from a prompt share one consistency score, this is a prompt-level metric used as a multiplier in the FRANQ score.

**How to modify:**
- **Change number of samples:** Edit `NUM_CONSISTENCY_SAMPLES` in `config.py` (3 is sufficient; 5 is more reliable but 67% slower)
- **Change temperature:** Edit the `temperature` value in the `options` dict in `consistency.py`. Higher temperature = more diversity = lower consistency scores (more informative).

---

### 6.9 `prompt_metrics.py`

**What it does:** Computes all prompt-level aggregated metrics from the per-claim scores.

**Functions:**

```python
grounding_score(faith_scores)
# Mean faithfulness across all claims. High = response is well-grounded in context.

hallucination_rate(faith_scores, conf_scores)
# Fraction of claims where BOTH faithfulness < 0.5 AND confidence < 0.4.
# These are the most suspicious claims.

confidence_calibration(conf_scores, faith_scores)
# Pearson correlation between confidence and faithfulness across claims.
# High positive value = model is confident when it's correct, uncertain when it's wrong.
# Range: [-1, 1]

relevance_score(prompt, response, embedder)
# Cosine similarity between prompt embedding and response embedding.
# Measures: did the model actually answer the question asked?

completeness_score(response, ground_truth, embedder)
# Cosine similarity between response and ground truth (if available).
# Returns None if no ground truth for this prompt.

robustness_score(prompt, context_docs, generate_fn, embedder)
# Paraphrases the prompt, re-generates, compares the two responses.
# High = stable output regardless of how the question is phrased.
# FAST_MODE=True returns None (skips the extra calls).
```

**How to modify:**
- **Change hallucination definition:** Edit the condition in `hallucination_rate()`. E.g. use only faithfulness (ignore confidence): `if f < FAITHFULNESS_THRESHOLD`
- **Add a new metric:** Define a new function here. Then add it to the `prompt_result` dict in `pipeline.py`.

---

### 6.10 `scorer.py`

**What it does:** Computes the FRANQ composite claim score and the 4-quadrant error taxonomy.

**FRANQ Score formula:**
```
base  = faithfulness × confidence  +  (1 - faithfulness) × parametric
base *= max(0, 1 - contradiction)
base *= consistency
```

Intuition:
- If the claim is **faithful**: reward grounded confidence (`faithfulness × confidence`)
- If the claim is **not faithful**: fall back to model's parametric knowledge (`parametric`)
- Penalise for contradictions
- Scale by consistency (unstable models get lower scores)

**Error taxonomy — `classify_claim(faithfulness, parametric)`:**

| Faithfulness | Parametric | Label |
|---|---|---|
| ≥ 0.5 | ≥ 0.4 | ✅ Grounded Correct |
| ≥ 0.5 | < 0.4 | ⚠️ Grounded Incorrect |
| < 0.5 | ≥ 0.4 | 🤔 Lucky Correct |
| < 0.5 | < 0.4 | ❌ Hallucination |

**How to modify:**
- **Change thresholds:** Edit `FAITHFULNESS_THRESHOLD` and `CONFIDENCE_THRESHOLD` in `config.py`
- **Change FRANQ weights:** Edit the formula directly in `franq_score()`. E.g. to weight contradiction more heavily: `base *= (1 - contradiction) ** 2`
- **Add more taxonomy categories:** Add more conditions in `classify_claim()`. E.g. split ❌ Hallucination into "Confident Hallucination" vs "Uncertain Hallucination" based on confidence score.

---

## 7. `pipeline.py` — End-to-End Runner

This is the main script that ties every module together. It:
1. Loads all models (retriever, NLI, embedder)
2. Reads `prompts.json` and `ground_truth.json`
3. For each prompt: retrieves → generates → extracts → scores → aggregates
4. Writes `outputs/results.json`
5. Prints a summary to the terminal

**CLI flags:**

| Flag | Effect |
|---|---|
| `--mini` | Only run first 5 prompts |
| `--fast` | Force `FAST_MODE=True` for this run |
| `--full` | Force `FAST_MODE=False` for this run |

**How to modify:**
- **Run a subset of prompt types:** Add a filter after loading prompts:
  ```python
  prompts = [p for p in prompts if p.get("type") == "factual"]
  ```
- **Resume from a checkpoint:** Save intermediate results and skip already-processed prompt IDs
- **Add a new metric column:** Compute the metric in the loop and add it to `prompt_result` dict. It will automatically appear in `results.json` and the dashboard.
- **Run in parallel:** Use `concurrent.futures.ThreadPoolExecutor` around the per-prompt loop (note: NLI and FAISS are not thread-safe; load separate instances per thread)

---

## 8. `dashboard.py` — Streamlit UI

Loads `outputs/results.json` and presents:

| Section | What it shows |
|---|---|
| Overall Summary | 5 metrics: Model Score, Avg Hallucination Rate, Avg Grounding, Avg Consistency, Avg Relevance |
| Scores by Prompt Type | Factual vs Reasoning vs Ambiguous breakdown |
| Per-Prompt Table | All 20 prompts with all metric columns |
| FRANQ Score Chart | Bar chart of FRANQ score per prompt |
| Error Taxonomy Chart | Bar chart of claim classifications across all prompts |
| Prompt Drill-Down | Select a prompt → see response, context docs, claim table, metric summary |

**How to run:**
```powershell
Set-Location "C:\Users\91930\Desktop\GENAI Final Project\llm_eval_framework"
& "C:\Users\91930\anaconda3\envs\gpu\Scripts\streamlit.exe" run dashboard.py
```

**How to modify:**
- **Add a new chart:** Use `st.line_chart()`, `st.scatter_chart()`, or Altair via `st.altair_chart()` after loading the dataframe
- **Show prompt text in table:** Add `"Prompt": p["prompt"]` to the row dict in the `rows` list
- **Filter by prompt type:** Add a sidebar multiselect: `types = st.sidebar.multiselect("Types", ["factual", "reasoning", "ambiguous"])` then filter `df`
- **Compare two result files:** Add a file uploader via `st.file_uploader()` and load a second results JSON for side-by-side comparison

---

## 9. Output Format — `results.json`

Full structure:

```json
{
  "model_score": 0.612,           // average FRANQ score across all prompts
  "fast_mode": true,
  "generator_model": "mistral",
  "claim_extractor_model": "qwen2.5",
  "prompts": [
    {
      "id": "p001",
      "type": "factual",
      "prompt": "What was ENIAC...",
      "response": "ENIAC was the first general-purpose...",
      "context": ["doc text 1...", "doc text 2...", "doc text 3..."],

      // ── Prompt-level metrics ──
      "prompt_score": 0.74,           // avg FRANQ score across claims
      "grounding_score": 0.81,        // avg faithfulness
      "hallucination_rate": 0.08,     // fraction of low-faith + low-conf claims
      "calibration": 0.65,            // Pearson correlation(confidence, faithfulness)
      "consistency_score": 1.0,       // avg pairwise similarity of re-samples (1.0 in FAST_MODE)
      "relevance_score": 0.78,        // cosine sim(prompt, response)
      "completeness_score": 0.82,     // cosine sim(response, ground_truth) or null
      "robustness_score": null,       // cosine sim(original, paraphrased response) or null in FAST_MODE

      "claims": [
        {
          "text": "ENIAC was completed in 1945.",
          "faithfulness": 0.91,
          "confidence": 0.67,
          "parametric": 0.5,          // 0.5 in FAST_MODE
          "contradiction": 0.02,
          "consistency": 1.0,
          "franq_score": 0.58,
          "classification": "✅ Grounded Correct"
        }
        // ... more claims
      ]
    }
    // ... more prompts
  ]
}
```

---

## 10. How to Modify — Common Scenarios

### Switch to a different Ollama model

Edit `config.py`:
```python
GENERATOR_MODEL = "llama3"           # or "phi4", "deepseek-r1:7b", etc.
CLAIM_EXTRACTOR_MODEL = "qwen2.5"    # keep qwen2.5 for best JSON output
```

Make sure you have pulled the model first: `ollama pull llama3`

---

### Add your own knowledge base documents

1. Add `.txt` files to `data/knowledge_base/`
2. No code changes needed — the retriever auto-discovers all `.txt` files
3. Delete `outputs/results.json` (old results are now stale)
4. Re-run the pipeline

---

### Add your own prompts

Edit `data/prompts.json`. Each entry needs:
- A unique `"id"` (e.g. `"p021"`)
- A `"prompt"` string
- A `"type"` of `"factual"`, `"reasoning"`, or `"ambiguous"`

Optionally add a reference answer to `data/ground_truth.json` using the same `id`.

---

### Use Gemini API instead of Ollama

In `modules/generator.py`, replace the function body:

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_answer(prompt, context_docs, with_context=True, model=None):
    m = model or "gemini-1.5-flash"
    context_str = "\n\n".join(context_docs) if (with_context and context_docs) else ""
    user_msg = f"Context:\n{context_str}\n\nQuestion: {prompt}" if with_context else f"Question: {prompt}"

    gemini = genai.GenerativeModel(m)
    response = gemini.generate_content(user_msg)
    return {
        "response": response.text,
        "token_logprobs": []   # Gemini does not expose logprobs → confidence_score returns 0.0
    }
```

Add `GEMINI_API_KEY=your_key_here` to `.env`.
Do the same in `claim_extractor.py` replacing the `ollama.Client().chat()` call.

---

### Enable full metrics (parametric + consistency + robustness)

In `config.py`:
```python
FAST_MODE = False
```

Or run with flag: `python pipeline.py --full`

Expect ~20–40 minutes for 20 prompts on local hardware.

---

### Change the scoring thresholds

In `config.py`:
```python
FAITHFULNESS_THRESHOLD = 0.65    # stricter — fewer claims labelled ✅
CONFIDENCE_THRESHOLD = 0.3       # looser — more claims considered "confident"
```

The new thresholds take effect on the next pipeline run.

---

## 11. Models Reference

### Ollama Models

| Model | Pull Command | Size | Best For |
|---|---|---|---|
| `mistral` | `ollama pull mistral` | 4.7 GB | Default generator, good general-purpose |
| `qwen2.5` | `ollama pull qwen2.5` | 4.7 GB | Claim extraction (excellent JSON mode) |
| `gemma3:4b` | `ollama pull gemma3:4b` | 3.3 GB | Consistency sampling (fast) |
| `llama3` | `ollama pull llama3` | 4.7 GB | Good alternative to mistral |
| `phi4` | `ollama pull phi4` | 9.1 GB | Best quality on 16 GB RAM |
| `deepseek-r1:7b` | `ollama pull deepseek-r1:7b` | 4.9 GB | Strong reasoning prompts |
| `gemma3:1b` | `ollama pull gemma3:1b` | 0.8 GB | Fastest option, lower quality |

### HuggingFace Models (auto-downloaded on first run)

| Model | Size | Used By |
|---|---|---|
| `all-MiniLM-L6-v2` | ~90 MB | Retriever, relevance, completeness, robustness |
| `cross-encoder/nli-deberta-v3-base` | ~500 MB | faithfulness.py, contradiction.py |

---

## 12. Troubleshooting

### `ConnectionError: ollama` or `Failed to connect`
- Ollama is not running. Start it: open the Ollama app (system tray) or run `ollama serve` in a terminal.
- Check `OLLAMA_HOST` in `config.py` matches where Ollama is running.

### `model not found` error
- The model hasn't been pulled yet. Run `ollama pull mistral` (or `qwen2.5`, `gemma3:4b`).

### `extract_claims()` returns `[]`
- The claim extractor model produced non-JSON output.
- Try switching `CLAIM_EXTRACTOR_MODEL` to `"qwen2.5"` (it has the best JSON adherence).
- The pipeline has a fallback: if no claims are extracted, it treats the full response as one claim.

### NLI model download is slow
- `cross-encoder/nli-deberta-v3-base` is ~500 MB and downloads on first run.
- It is cached in `C:\Users\91930\.cache\huggingface\hub` — subsequent runs are instant.
- To use a smaller/faster NLI: change `NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"` in `config.py`.

### Low faithfulness scores on obviously correct claims
- The NLI model context window is capped at 1500 characters (to avoid token overflow). Long documents may be truncated, causing the relevant passage to be cut off.
- Fix: increase the `[:1500]` slice in `faithfulness.py` and `contradiction.py`, or score against each document individually and take the max.

### Dashboard shows "No results file found"
- Run `python pipeline.py --mini` first to generate `outputs/results.json`.

### `streamlit: command not found`
- Use the full path: `& "C:\Users\91930\anaconda3\envs\gpu\Scripts\streamlit.exe" run dashboard.py`

### `ModuleNotFoundError: No module named 'config'`
- The pipeline must be run from inside the `llm_eval_framework/` folder.
- Fix: `Set-Location "C:\Users\91930\Desktop\GENAI Final Project\llm_eval_framework"` before running.
