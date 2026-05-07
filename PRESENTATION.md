# Prompt-wise LLM Evaluation Framework for Fixed-Knowledge RAG Systems
### GenAI Final Project Presentation

---

## Slide 1 — Title & Team

**Project Title:**  
> Prompt-wise LLM Evaluation Framework for Fixed-Knowledge RAG Systems

**Subtitle:**  
> Atomic Claim-Level Hallucination Detection, Faithfulness Scoring, and Error Taxonomy for Local Language Models

**Key Pitch (say this in opening):**
> "Most LLM evaluations score an entire response with a single number. We broke every answer into atomic facts and scored *each one individually* — so we can tell you not just *how wrong* a model was, but *exactly which sentence* was hallucinated and *why*."

---

## Slide 2 — Motivation & Problem Statement

### The Problem with Current LLM Evaluation

| Approach | What it misses |
|---|---|
| BLEU / ROUGE | Only surface-level word overlap — misses meaning |
| Human rating | Expensive, non-reproducible, not scalable |
| Whole-response LLM-as-judge | Can't pinpoint *which* part of the answer is wrong |
| Simple embedding similarity | Doesn't detect internal contradictions |

### Why This Matters
- Deployed RAG systems **hallucinate silently** — the model sounds confident even when wrong
- A 10-sentence answer may have 9 correct sentences and 1 dangerous hallucination — response-level scoring **hides it**
- You need **per-claim accountability** to safely deploy LLMs in high-stakes domains

### Our Insight
> Decompose → Score → Classify  
> Every response → atomic claims → individual metric scores → error taxonomy label

---

## Slide 3 — System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   LLM Evaluation Framework                       │
│                                                                  │
│  Prompt  ──►  [FAISS Retriever]  ──►  Top-K Docs               │
│                                          │                       │
│                                          ▼                       │
│                               [Ollama Generator]                 │
│                               (mistral / RAG answer)            │
│                                          │                       │
│                         ┌────────────────┼──────────────────┐   │
│                         │                │                  │   │
│                         ▼                ▼                  ▼   │
│              [Claim Extractor]  [Context-Free Gen]  [Token LogP] │
│              (qwen2.5, JSON)    (parametric check)  (confidence) │
│                         │                                        │
│                         ▼   (for each atomic claim)             │
│           ┌─────────────────────────────────────┐               │
│           │         PER-CLAIM SCORING            │               │
│           │  ├── faithfulness_score (NLI)        │               │
│           │  ├── contradiction_score (NLI)       │               │
│           │  ├── confidence_score (log-prob)     │               │
│           │  ├── parametric_score (NLI)          │               │
│           │  └── franq_score (composite)         │               │
│           └─────────────────────────────────────┘               │
│                         │                                        │
│                         ▼  (aggregated per prompt)               │
│           ┌─────────────────────────────────────┐               │
│           │      PROMPT-LEVEL METRICS            │               │
│           │  ├── grounding_score                 │               │
│           │  ├── hallucination_rate              │               │
│           │  ├── calibration                     │               │
│           │  ├── relevance_score                 │               │
│           │  ├── completeness_score              │               │
│           │  └── robustness_score                │               │
│           └─────────────────────────────────────┘               │
│                         │                                        │
│                         ▼                                        │
│              [Streamlit Dashboard]  ◄── outputs/results.json    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Slide 4 — Tech Stack

| Layer | Component | Why Chosen |
|---|---|---|
| LLM (generator) | **Mistral 7B** via Ollama | Strong instruction-following, free, local |
| LLM (extractor) | **Qwen 2.5** via Ollama | Best JSON-mode adherence of tested models |
| LLM (consistency) | **Gemma 3 4B** via Ollama | Lightweight, fast for re-sampling |
| NLI Model | **DeBERTa-v3-base CrossEncoder** | State-of-art NLI, GPU-accelerated |
| Embeddings | **all-MiniLM-L6-v2** | Fast, 384-dim, good semantic similarity |
| Vector Store | **FAISS** (local) | Zero-latency retrieval, no API needed |
| Dashboard | **Streamlit** | Rapid interactive visualization |
| Runtime | Python 3.10, CUDA GPU | Local, private, reproducible |

**Key design principle:** 100% local — no OpenAI, no external APIs, no data leaving the machine.

---

## Slide 5 — The RAG Pipeline (Step-by-Step)

### Step 1: Retrieval
- Knowledge base: **10 `.txt` documents** covering diverse topics
- Embedded with `all-MiniLM-L6-v2` → stored in **FAISS index**
- Per query: **top-K=3** most semantically relevant documents retrieved

### Step 2: Generation
```
Prompt + Top-3 docs → Ollama (mistral) → Response + token log-probabilities
```
- Two generations per prompt:
  - **With context** → RAG answer (main evaluation target)
  - **Without context** → context-free answer (for parametric scoring)

### Step 3: Claim Extraction
```python
# qwen2.5 in JSON mode decomposes:
"ENIAC was built in 1945 and weighed 30 tons"
→ ["ENIAC was built in 1945", "ENIAC weighed 30 tons"]
```
- Each claim is a single verifiable factual statement
- Enables independent scoring — one wrong claim doesn't contaminate others

### Step 4: Per-Claim Scoring
- **5 scores computed per claim** (see Slide 6–11)
- Token log-probabilities aligned to each claim's character span

### Step 5: Prompt-Level Aggregation
- **7 prompt-level metrics** aggregated from claim scores (see Slide 12–15)

### Step 6: Error Taxonomy Classification
- Each claim gets a **4-quadrant label** (see Slide 16)

---

## Slide 6 — Metric 1: Faithfulness Score

**File:** `modules/faithfulness.py`  
**Range:** `[0, 1]` — higher is better  
**Always computed** (even in fast mode)

### What it measures
> How well the claim is *supported* by the retrieved context documents.

### How it works: Per-Document NLI

The model is **not** given all documents at once (which causes dilution). Each document is scored independently:

$$\text{faithfulness}(c) = \max_{d \in \text{docs}} \; P_{\text{NLI}}(\text{doc}_d \;\text{entails}\; c)$$

**Why max and not average?**  
If one document strongly supports a claim, that's all that matters. Averaging would penalise a well-supported claim just because irrelevant documents score low.

### NLI Model
- `cross-encoder/nli-deberta-v3-base` (HuggingFace)
- CrossEncoder outputs 3 logits: `[contradiction=0, entailment=1, neutral=2]`
- Softmax → entailment probability used here

| Score | Meaning |
|---|---|
| > 0.7 | Claim is well-supported by context |
| 0.4 – 0.7 | Partial support |
| < 0.4 | Context does not support this claim |

---

## Slide 7 — Metric 2: Contradiction Score

**File:** `modules/contradiction.py`  
**Range:** `[0, 1]` — lower is better  
**Always computed**

### What it measures
> Whether any retrieved document *actively contradicts* the claim.

$$\text{contradiction}(c) = \max_{d \in \text{docs}} \; P_{\text{NLI}}(\text{doc}_d \;\text{contradicts}\; c)$$

Same NLI model, same singleton — only the output index differs (`index 0 = contradiction`).

### Combined Faithfulness + Contradiction Interpretation

| Faithfulness | Contradiction | Diagnosis |
|---|---|---|
| High | Low | ✅ Claim supported, no conflict |
| Low | Low | ⚪ Context is silent on this claim |
| Low | High | ❌ Claim directly conflicts with context |
| High | High | ⚠️ Ambiguous/conflicting context |

### Why this is separate from faithfulness
A claim can have *low faithfulness* (not supported) without being *contradicted* — the context just doesn't mention it. These are two different failure modes requiring different remedies.

---

## Slide 8 — Metric 3: Confidence Score (Log-Probability)

**File:** `modules/confidence.py`  
**Range:** `(0, 1]` — higher is better  
**Always computed**

### What it measures
> How certain the model was when it generated the tokens that form this claim.

### Formula: Geometric Mean of Token Probabilities

$$\text{confidence}(c) = \exp\!\left(\frac{1}{N} \sum_{t=1}^{N} \log P(t)\right)$$

This is equivalent to the **inverse perplexity** of the claim tokens. A model that generates every token with certainty scores 1.0.

### Per-Claim Token Alignment (Key Innovation)

Without alignment, every claim in a response gets the *same* score (whole-response average). With alignment:

1. Ollama returns `token` + `log-probability` for every generated token
2. Tokens are concatenated to reconstruct the full response
3. The claim's character span is located in the reconstruction
4. Only tokens overlapping that span contribute to the score

```
Response tokens: ["The", " ENIAC", " was", " built", " in", " 1945", " and", " weighed", ...]
Claim 1 span:     ←────────── "ENIAC was built in 1945" ──────────→
Claim 2 span:                                             ←──── "weighed 30 tons" ────→
```

This means a hedged phrase like *"it may have been..."* correctly gets **lower confidence** than a direct factual statement.

---

## Slide 9 — Metric 4: Parametric Score

**File:** `modules/parametric.py`  
**Range:** `[0, 1]`  
**Skipped in fast mode (returns 0.5 neutral)**

### What it measures
> Does the model's *internal* knowledge (without any context) also support this claim?

Tests whether the model "already knew" the answer independently of RAG.

$$\text{parametric}(c) = P_{\text{NLI}}(\text{context-free answer entails claim})$$

### Faithfulness vs. Parametric — The 4 Cases

| Faithfulness | Parametric | Diagnosis |
|---|---|---|
| High | High | ✅ Model already knew this — well-covered topic |
| High | Low | 🔵 Claim came *only* from retrieved context (RAG working correctly) |
| Low | High | 🟡 Model confident but context doesn't confirm — possible overconfidence |
| Low | Low | ❌ **Hallucination** — neither context nor internal knowledge supports it |

### Why this matters
This distinguishes *hallucination* from *context-grounded novelty*. A RAG system that retrieves rare facts the model doesn't know should score high faithfulness, low parametric — and that is **correct behaviour**, not a failure.

---

## Slide 10 — Metric 5: Consistency Score

**File:** `modules/consistency.py`  
**Range:** `[0, 1]` — higher is better  
**Skipped in fast mode (returns 1.0)**

### What it measures
> How stable is the model's answer across multiple independent samplings of the same question?

### How it works

1. Generate `k=3` responses at `temperature=0.7` using Gemma 3 4B
2. Encode each response with `all-MiniLM-L6-v2`
3. Compute average pairwise cosine similarity:

$$\text{consistency} = \frac{1}{\binom{k}{2}} \sum_{i < j} \cos(\vec{v}_i, \vec{v}_j)$$

### Interpretation
- **Score near 1.0**: Model always says the same thing — reliable
- **Score near 0.5**: Answers vary significantly — model is uncertain or topic-sensitive
- Consistency is **prompt-level** (one score per prompt, same value applied to all claims of that prompt)

---

## Slide 11 — Metric 6: FRANQ Score (Composite)

**File:** `modules/scorer.py`  
**Range:** `[0, 1]` — higher is better  
**Always computed**

### What it measures
> A single quality score for one claim that combines all the metrics above.

### Formula (3-step)

**Step 1 — Base score (faithfulness-weighted blend):**
$$\text{base} = \text{faith} \times \text{conf} + (1 - \text{faith}) \times \text{param}$$

- When faithfulness is **high**: reward confident grounded answers (`faith × conf`)
- When faithfulness is **low**: fall back to parametric knowledge (`(1-faith) × param`)

**Step 2 — Contradiction penalty:**
$$\text{base} = \text{base} \times \max(0,\; 1 - \text{contra})$$

**Step 3 — Consistency scaling:**
$$\text{FRANQ} = \text{base} \times \text{consist}$$

### Why "FRANQ"
**F**aithfulness — **R**eliability — **A**ccuracy — **N**LI — **Q**uality

Inspired by RAFT/RAGAS literature but extended with token-level confidence and consistency.

---

## Slide 12 — Prompt-Level Metric 1: Grounding Score

**File:** `modules/prompt_metrics.py`  
**Range:** `[0, 1]`

$$\text{grounding} = \frac{1}{|C|} \sum_{c \in C} \text{faithfulness}(c)$$

Average faithfulness across all claims. Answers: *"What fraction of this response was grounded in the knowledge base?"*

---

## Slide 13 — Prompt-Level Metric 2: Hallucination Rate

**File:** `modules/prompt_metrics.py`  
**Range:** `[0, 1]` — lower is better

$$\text{hallucination\_rate} = \frac{|\{c : \text{faith}(c) < \tau_f \;\wedge\; \text{conf}(c) < \tau_c\}|}{|C|}$$

Fraction of claims that have **both** low faithfulness (not in context) **and** low confidence (model uncertain). These are the highest-risk hallucinations.

**Important distinction:**
- Low faith + high confidence → "confidently wrong" (dangerous)
- Low faith + low confidence → "uncertain unsupported" (less dangerous — model is hedging)

Thresholds: `FAITHFULNESS_THRESHOLD = 0.5`, `CONFIDENCE_THRESHOLD = 0.4`

---

## Slide 14 — Prompt-Level Metric 3: Confidence Calibration

**File:** `modules/prompt_metrics.py`  
**Range:** `[-1, 1]`

$$\text{calibration} = \text{Pearson}\!\left(\text{confidence}_c,\; \text{faithfulness}_c\right)$$

Pearson correlation between confidence and faithfulness across all claims.

| Value | Meaning |
|---|---|
| Near +1 | ✅ Well-calibrated: model is confident on grounded claims, uncertain on unsupported ones |
| Near 0 | Confidence is random with respect to groundedness |
| Negative | ❌ Miscalibrated: model is *more* confident on *unsupported* claims — overconfident hallucinator |

This is one of the most diagnostic metrics for unsafe deployment scenarios.

**Important caveat — low-variance prompts:**  
When all claims in a prompt have very similar faithfulness scores (e.g. all near 0.99), the Pearson denominator variance approaches zero and the resulting calibration value becomes numerically unstable — the sign is noise, not signal. Calibration is only meaningful when there is real spread in faithfulness across claims within the prompt.

---

## Slide 15 — Prompt-Level Metrics 4–7

### Relevance Score
$$\text{relevance} = \cos\!\left(\text{embed}(\text{prompt}),\; \text{embed}(\text{response})\right)$$
Does the response actually answer the question asked?

### Completeness Score
$$\text{completeness} = \cos\!\left(\text{embed}(\text{response}),\; \text{embed}(\text{ground\_truth})\right)$$
How much of the expected answer is covered? Requires ground truth entry.

### Robustness Score *(full mode only)*
$$\text{robustness} = \cos\!\left(\text{embed}(\text{response}_\text{orig}),\; \text{embed}(\text{response}_\text{paraphrased})\right)$$
Is the model brittle to phrasing changes? Paraphrases the prompt via Ollama then re-evaluates.

### Prompt Score
$$\text{prompt\_score} = \frac{1}{|C|}\sum_{c \in C}\text{FRANQ}(c)$$
Final single score for the entire prompt response.

---

## Slide 16 — Error Taxonomy (Claim Classification)

**File:** `modules/scorer.py`

Every claim is placed in exactly one of 4 quadrants based on two thresholds:
- **Faithfulness ≥ 0.5** → retrieved context *supports* the claim
- **Parametric ≥ 0.4** → model's *own internal knowledge* supports the claim

```
                    PARAMETRIC
                  Low        High
              ┌──────────┬──────────────┐
  FAITHFULNESS│          │              │
         High │⚠️ Grounded│ ✅ Grounded  │
              │ Incorrect│   Correct    │
              ├──────────┼──────────────┤
          Low │ ❌ Halluc.│ 🤔 Lucky     │
              │          │   Correct    │
              └──────────┴──────────────┘
```

---

### ✅ Grounded Correct — Faithfulness HIGH + Parametric HIGH

The context supports the claim **and** the model already knew it independently. The gold standard — the model said something true, the knowledge base confirms it, and the model would have said the same thing even without RAG.

> *Example: "James Watson and Francis Crick determined the structure of DNA"* — famous enough that the model knows it, and it appears word-for-word in the document.

---

### ⚠️ Grounded Incorrect — Faithfulness HIGH + Parametric LOW

The context supports the claim, but the model didn't know it without the retrieved documents. **Despite the label name, this is not actually wrong** — it means RAG is doing its job by providing knowledge the model lacks on its own. Think of it as *"context-dependent"*.

> *Example: a specific measurement or obscure date that appears in your knowledge base but isn't in the model's training distribution.*

---

### 🤔 Lucky Correct — Faithfulness LOW + Parametric HIGH

The model said something from its own internal knowledge, but the retrieved context doesn't confirm it. This is ambiguous — it could be:
- A true fact that just isn't covered in your 10 documents (model is right, retrieval missed it)
- The model hallucinating confidently from noisy training data

Named "lucky" because you cannot tell from metrics alone which case it is.

---

### ❌ Hallucination — Faithfulness LOW + Parametric LOW

Neither the context nor the model's internal knowledge supports the claim. The clearest hallucination signal — the model made something up, it's not in the documents, and even the context-free model doesn't say it.

> *Example from results: "Pavlov would sometimes stop ringing the bell for several days or weeks" — this specific procedural detail is not in the psychology document and the model doesn't know it without context either.*

---

### Important: FAST_MODE collapses the taxonomy

With `FAST_MODE=True`, parametric always returns `0.5` (above the 0.4 threshold), so parametric is always treated as "HIGH". This reduces the 4 labels to just 2:

| Faithfulness | Label in FAST_MODE |
|---|---|
| ≥ 0.5 | ✅ Grounded Correct |
| < 0.5 | 🤔 Lucky Correct |

You will **never** see ⚠️ Grounded Incorrect or ❌ Hallucination in fast mode results. Those labels only appear with `--full` where real parametric scoring runs.

---

## Slide 17 — Complete Metrics Summary Table

| # | Metric | Level | Range | Better | Fast Mode |
|---|---|---|---|---|---|
| 1 | Faithfulness Score | Claim | [0, 1] | Higher | ✅ Always |
| 2 | Contradiction Score | Claim | [0, 1] | Lower | ✅ Always |
| 3 | Confidence Score | Claim | (0, 1] | Higher | ✅ Always |
| 4 | Parametric Score | Claim | [0, 1] | Higher | ⏩ 0.5 neutral |
| 5 | Consistency Score | Claim/Prompt | [0, 1] | Higher | ⏩ 1.0 neutral |
| 6 | FRANQ Score | Claim | [0, 1] | Higher | ✅ Always |
| 7 | Error Taxonomy Label | Claim | 4 classes | — | ✅ Always |
| 8 | Prompt Score | Prompt | [0, 1] | Higher | ✅ Always |
| 9 | Grounding Score | Prompt | [0, 1] | Higher | ✅ Always |
| 10 | Hallucination Rate | Prompt | [0, 1] | Lower | ✅ Always |
| 11 | Calibration | Prompt | [-1, 1] | Near +1 | ✅ Always |
| 12 | Relevance Score | Prompt | [0, 1] | Higher | ✅ Always |
| 13 | Completeness Score | Prompt | [0, 1] | Higher | ✅ Always |
| 14 | Robustness Score | Prompt | [0, 1] | Higher | ⏩ null |
| 15 | Model Score | Dataset | [0, 1] | Higher | ✅ Always |

---

## Slide 18 — Models Used

| Model | Role | Type | Size |
|---|---|---|---|
| **Mistral 7B** | RAG answer generator | Ollama (local) | 4.7 GB |
| **Qwen 2.5 7B** | Atomic claim extractor | Ollama (local, JSON mode) | 4.7 GB |
| **Gemma 3 4B** | Consistency re-sampler | Ollama (local) | 3.3 GB |
| **DeBERTa-v3-base** | NLI (faithfulness + contradiction + parametric) | HuggingFace CrossEncoder | ~500 MB |
| **all-MiniLM-L6-v2** | Embeddings (retrieval + similarity) | sentence-transformers | ~90 MB |

**Why all-local?**
- No API costs, no rate limits, no data privacy concerns
- Reproducible results (no model updates breaking evaluations)
- Runs on consumer GPU (tested on 8GB VRAM)

---

## Slide 19 — Knowledge Bases & Datasets

### Dataset 1 (data/)
10 general knowledge documents covering:
History of Computers, Internet & Networking, AI, World Geography, Climate & Weather, Human Biology, Space Exploration, Economics, World History, Mathematics

**20 evaluation prompts** spanning: factual, reasoning, multi-hop, open-ended types

### Dataset 2 (data2/)
10 STEM-focused documents covering:
Genetics, Chemistry, Physics, Psychology, Nutrition, Astronomy, Programming Languages, Renewable Energy, Medicine History, Philosophy

**Includes:**
- `hallucination_test_prompts_dataset.md` — prompts specifically designed to induce hallucination
- `truthful_qa_evaluation_questions.md` — TruthfulQA-style adversarial questions

**Why two datasets?**  
Cross-dataset evaluation shows whether framework metrics generalise across domains — not overfit to one knowledge base.

---

## Slide 20 — Pipeline Run Modes

| Mode | Flag | Prompts | FAST_MODE | Extra LLM Calls | Time |
|---|---|---|---|---|---|
| Mini | `--mini` | 5 | True | None | ~1–2 min |
| Default | *(none)* | 20 | True | None | ~5–8 min |
| Fast | `--fast` | 20 | True (forced) | None | ~5–8 min |
| Full | `--full` | 20 | False | parametric + consistency + robustness | ~20–40 min |

**FAST_MODE=True skips:** parametric scoring, consistency sampling, robustness scoring → saves ~80% of LLM calls.

```powershell
# Run from llm_eval_framework/ folder
$py = "C:\Users\91930\anaconda3\envs\gpu\python.exe"

& $py pipeline.py --mini     # test 5 prompts
& $py pipeline.py            # full evaluation
& $py pipeline.py --full     # all metrics
& $py pipeline.py --data data2  # second dataset

# Launch dashboard
streamlit run dashboard.py
```

---

## Slide 21 — Streamlit Dashboard Features

The interactive dashboard (`dashboard.py`) provides:

1. **File Selector** — compare multiple result runs side-by-side
2. **Summary Cards** — model score, avg FRANQ, avg hallucination rate, avg calibration
3. **Prompt Table** — sortable table of all prompts with all 9+ metric columns
4. **Per-Prompt Drill-Down** — select any prompt to see:
   - The full generated response
   - Each extracted claim with its individual scores
   - Error taxonomy badge per claim
5. **Score Distribution Charts** — histograms of FRANQ, faithfulness, hallucination rate
6. **Claim Classification Breakdown** — pie chart of ✅/⚠️/🤔/❌ labels

---

## Slide 22 — Key Differentiators vs. Baseline Approaches

| Feature | Naive Eval | RAGAS | **Our Framework** |
|---|---|---|---|
| Scoring granularity | Response-level | Response-level | **Claim-level** |
| Confidence from model | ✗ | ✗ | **✅ Token log-probs** |
| Per-claim token alignment | ✗ | ✗ | **✅ Character-span alignment** |
| Hallucination vs. silence | Not separated | Not separated | **✅ Separated** |
| Parametric vs. context | ✗ | ✗ | **✅ Both scored** |
| Error taxonomy | ✗ | Partial | **✅ 4-quadrant classification** |
| Local-only inference | Sometimes | ✗ (needs OpenAI) | **✅ 100% local** |
| Calibration measurement | ✗ | ✗ | **✅ Pearson corr** |
| Multi-dataset support | ✗ | ✗ | **✅ data/ and data2/** |

---

## Slide 23 — Example Output (results.json structure)

```json
{
  "model": "mistral",
  "model_score": 0.71,
  "prompts": [
    {
      "id": "p01",
      "type": "factual",
      "prompt": "When was ENIAC built?",
      "response": "ENIAC was completed in 1945 and was the first general-purpose computer.",
      "prompt_score": 0.74,
      "grounding_score": 0.81,
      "hallucination_rate": 0.0,
      "calibration": 0.62,
      "relevance_score": 0.89,
      "completeness_score": 0.76,
      "claims": [
        {
          "text": "ENIAC was completed in 1945",
          "faithfulness": 0.88,
          "contradiction": 0.04,
          "confidence": 0.73,
          "parametric": 0.91,
          "franq_score": 0.79,
          "label": "Grounded Correct"
        },
        {
          "text": "ENIAC was the first general-purpose computer",
          "faithfulness": 0.61,
          "contradiction": 0.12,
          "confidence": 0.68,
          "parametric": 0.55,
          "franq_score": 0.58,
          "label": "Grounded Correct"
        }
      ]
    }
  ]
}
```

---

## Slide 24 — Challenges & Solutions

| Challenge | Solution |
|---|---|
| NLI dilution when multiple docs concatenated | Per-document NLI scoring + max pooling |
| Confidence identical for all claims in a response | Character-span token alignment |
| qwen2.5 wraps JSON in markdown fences | JSON parser strips ` ```json ``` ` blocks with regex fallback |
| First run downloads 500 MB NLI model | Singleton cache — downloads once, reused across all claims |
| RAG retrieval on 10 docs is shallow | FAISS cosine similarity with sentence-transformer embeddings |
| parametric scoring doubles LLM calls | Skippable in FAST_MODE, returns 0.5 neutral (unbiased default) |
| Multi-model coordination (3 Ollama models) | Each configured independently in config.py, hot-swappable |

---

## Slide 25 — What Makes This Project Full-Marks Worthy

### Technical Depth
- **14+ distinct metrics** computed at two levels of granularity
- **NLI-based faithfulness** with mathematically motivated per-document max pooling
- **Token log-probability confidence** with per-claim character-span alignment
- **FRANQ composite scoring** with principled faithfulness-weighted blending
- **4-quadrant error taxonomy** separating hallucination from grounded novelty

### Engineering Quality
- All settings in one `config.py` — zero hardcoding elsewhere
- Singleton NLI model shared across modules — no double-loading
- `--mini`, `--fast`, `--full` pipeline modes for different evaluation budgets
- Dual dataset support (`data/` vs `data2/`) with CLI `--data` flag
- Streamlit dashboard with multi-file comparison

### Evaluation Rigour
- Two independent knowledge bases, one with adversarial hallucination prompts
- TruthfulQA-style evaluation questions for adversarial testing
- All metrics have mathematical formulas and interpretation tables

### Presentation Points to Emphasise
1. **"Why claim-level?"** — this is the core differentiator, explain it clearly
2. **"Per-document NLI, not concatenated"** — shows you understand NLI dilution
3. **"Token alignment for per-claim confidence"** — novel engineering detail
4. **"FRANQ formula derivation"** — walk through the 3 steps logically
5. **"Calibration metric"** — most evaluators haven't seen this; explain it carefully
6. **"Full mode vs. fast mode"** — shows awareness of real deployment constraints
7. **"100% local"** — highlight reproducibility and privacy as design goals

---

## Slide 26 — Q&A Preparation

**Q: Why not just use RAGAS or DeepEval?**  
A: Those require OpenAI/cloud APIs, score at response level, and don't provide per-token confidence or calibration. We built ours locally with better claim-level granularity.

**Q: Why DeBERTa for NLI?**  
A: `cross-encoder/nli-deberta-v3-base` is the strongest CrossEncoder on NLI benchmarks that fits on consumer GPU. It outperforms MiniLM variants by ~3–4% F1 on MNLI.

**Q: Why geometric mean for confidence?**  
A: Geometric mean of probabilities equals exp(mean log-prob) — this is equivalent to inverse perplexity, a standard information-theoretic measure of model certainty. Arithmetic mean would over-weight a few high-probability filler tokens.

**Q: What if claim extractor produces redundant claims?**  
A: Redundant claims get similar scores — this slightly inflates FRANQ but doesn't change classification labels. Future work: claim deduplication via cosine similarity.

**Q: Why not fine-tune the NLI model on domain-specific data?**  
A: The DeBERTa CrossEncoder generalises well out-of-box; fine-tuning on our 10-document knowledge base would risk overfitting and remove generalisability.

**Q: How does this scale to production?**  
A: FAST_MODE reduces LLM calls by ~80%. Further optimisation: batch NLI scoring (currently single-sample per call), async Ollama calls, claim caching.

---

## Slide 27 — Conclusion

### Summary
We built a **prompt-wise, claim-level LLM evaluation framework** that:
- Decomposes every LLM response into atomic facts
- Scores each fact on 6 metrics (faithfulness, contradiction, confidence, parametric, consistency, FRANQ)
- Aggregates into 7 prompt-level metrics including calibration and robustness
- Classifies each claim into a 4-label error taxonomy
- Visualises everything in an interactive Streamlit dashboard
- Runs entirely locally on consumer hardware

### Impact
This framework answers the question that basic LLM benchmarks cannot:  
**"Exactly which part of this answer is hallucinated, how confident was the model about it, and does the context support or contradict it?"**

---

## Appendix A — File Descriptions

| File | Purpose |
|---|---|
| `config.py` | All settings — models, thresholds, paths, modes |
| `pipeline.py` | End-to-end runner — orchestrates all modules |
| `dashboard.py` | Streamlit interactive UI |
| `modules/retriever.py` | FAISS-based document retrieval |
| `modules/generator.py` | Ollama LLM generation (with/without context) |
| `modules/claim_extractor.py` | qwen2.5 JSON-mode claim decomposition |
| `modules/faithfulness.py` | NLI entailment scoring |
| `modules/contradiction.py` | NLI contradiction scoring |
| `modules/confidence.py` | Token log-prob confidence + claim alignment |
| `modules/parametric.py` | Context-free knowledge scoring |
| `modules/consistency.py` | Multi-sample cosine similarity |
| `modules/scorer.py` | FRANQ formula + error taxonomy |
| `modules/prompt_metrics.py` | Prompt-level aggregations |
| `data/` | Dataset 1 — general knowledge, 20 prompts |
| `data2/` | Dataset 2 — STEM + adversarial hallucination prompts |

---

## Appendix B — Mathematical Foundations

### All Formulas in One Place

| Metric | Formula |
|---|---|
| Faithfulness | $\max_{d} P_\text{NLI}(d \to c)$ |
| Contradiction | $\max_{d} P_\text{NLI}(d \leftrightarrow\!\!\!\!\!\!\not\;\; c)$ |
| Confidence | $\exp\!\left(\frac{1}{N}\sum_t \log P(t)\right)$ |
| Parametric | $P_\text{NLI}(\text{cf-answer} \to c)$ |
| Consistency | $\frac{1}{\binom{k}{2}}\sum_{i<j}\cos(\vec{v}_i, \vec{v}_j)$ |
| FRANQ | $(\text{faith} \cdot \text{conf} + (1-\text{faith}) \cdot \text{param}) \cdot (1-\text{contra}) \cdot \text{consist}$ |
| Grounding | $\frac{1}{\|C\|}\sum_c \text{faith}(c)$ |
| Hallucination Rate | $\frac{\|\{c : \text{faith}(c)<\tau_f \wedge \text{conf}(c)<\tau_c\}\|}{\|C\|}$ |
| Calibration | $\text{Pearson}(\text{conf}_c, \text{faith}_c)$ |
| Relevance | $\cos(\text{emb}(q), \text{emb}(r))$ |
| Completeness | $\cos(\text{emb}(r), \text{emb}(\text{gt}))$ |
| Robustness | $\cos(\text{emb}(r_\text{orig}), \text{emb}(r_\text{para}))$ |
| Prompt Score | $\frac{1}{\|C\|}\sum_c \text{FRANQ}(c)$ |

---

*Generated for GenAI Final Project Presentation — LLM Eval Framework*
