# Metrics Reference

This document explains every metric computed by the framework — what it measures, how it is calculated, what the score range means, and how to interpret results.

Metrics fall into two levels:
- **Claim-level** — computed for each atomic claim extracted from a response
- **Prompt-level** — aggregated across all claims, or computed from the full response

---

## What is NLI?

**Natural Language Inference (NLI)** is the task of determining the logical relationship between two pieces of text:

- **Premise** — the reference text (in this framework: a retrieved context document)
- **Hypothesis** — the statement to test (in this framework: an extracted claim)

The NLI model outputs a probability for each of three classes:

| Class | Meaning |
|---|---|
| **Entailment** | The premise logically supports or implies the hypothesis |
| **Contradiction** | The premise directly conflicts with the hypothesis |
| **Neutral** | The premise neither supports nor contradicts the hypothesis |

**Why per-document scoring?**  
When multiple context documents are concatenated into one premise (e.g. "computers… world history… internet…"), the NLI model predicts NEUTRAL because no single topic dominates. Instead, this framework scores each document separately and takes the MAX, which finds the most supportive or most contradicting document — the semantically correct definition.

**Model used:** `cross-encoder/nli-deberta-v3-base` (HuggingFace, ~500 MB, runs on GPU)

---

## Claim-Level Metrics

These are computed independently for every atomic factual claim extracted from the response.

---

### 1. Faithfulness Score

**File:** `modules/faithfulness.py`  
**Range:** `[0, 1]` — higher is better  
**FAST_MODE:** always runs  

**What it measures:**  
How well the claim is supported by the retrieved context documents. Specifically: the maximum entailment probability across all retrieved documents.

**How it works:**
1. The NLI model is given each context document as the *premise* and the claim as the *hypothesis*
2. The entailment probability is extracted for each document
3. The maximum is returned — the most supportive document wins

$$\text{faithfulness} = \max_{d \in \text{docs}} P(\text{doc}_d \text{ entails claim})$$

**Interpretation:**
| Score | Meaning |
|---|---|
| `> 0.7` | Claim is well-supported by context |
| `0.4 – 0.7` | Partial support |
| `< 0.4` | Context does not support this claim |

---

### 2. Contradiction Score

**File:** `modules/contradiction.py`  
**Range:** `[0, 1]` — lower is better  
**FAST_MODE:** always runs  

**What it measures:**  
Whether any retrieved document actively contradicts the claim. High contradiction means the model said something that directly conflicts with the knowledge base.

**How it works:** Same pipeline as faithfulness but extracts the *contradiction* probability instead of entailment.

$$\text{contradiction} = \max_{d \in \text{docs}} P(\text{doc}_d \text{ contradicts claim})$$

**Interpretation combined with faithfulness:**
| Faithfulness | Contradiction | Diagnosis |
|---|---|---|
| High | Low | ✅ Claim supported, no conflict |
| Low | Low | Context is silent on this claim |
| Low | High | ❌ Claim conflicts with context |
| High | High | Unusual — ambiguous context |

---

### 3. Confidence Score

**File:** `modules/confidence.py`  
**Range:** `(0, 1]` — higher is better  
**FAST_MODE:** always runs  

**What it measures:**  
How certain the model was when generating the tokens that make up this claim. Derived from the token log-probabilities returned by Ollama.

**How it works (per-claim alignment):**
1. Ollama returns a log-probability for every generated token alongside the token string
2. The token strings are concatenated to reconstruct the full response text
3. The claim substring is located within that reconstructed text
4. Only the log-probabilities of tokens overlapping the claim span are used
5. Geometric mean is computed:

$$\text{confidence} = \exp\left(\frac{1}{N}\sum_{t=1}^{N}\log P(t)\right)$$

This is equivalent to inverse perplexity — a model that generates every token with probability 1 would score 1.0.

**Why per-claim rather than whole-response?**  
Without alignment, every claim in a response gets the identical score (the whole-response average). With alignment, a hedged claim ("it may have been…") correctly gets lower confidence than a direct factual statement.

**Falls back** to whole-response confidence if the claim can't be located in the token stream (e.g. claim was paraphrased during extraction).

---

### 4. Parametric Score

**File:** `modules/parametric.py`  
**Range:** `[0, 1]`  
**FAST_MODE:** returns `0.5` (neutral) — skips extra LLM call  

**What it measures:**  
Whether the model's *internal* knowledge (without any retrieved context) also supports the claim. Tests if the model "already knew" the answer independently of RAG.

**How it works:**
1. Re-generates an answer for the same prompt with **no context documents**
2. Runs NLI: does the context-free answer entail the claim?

$$\text{parametric} = P(\text{context-free answer entails claim})$$

**Interpretation:**
| Faithfulness | Parametric | Diagnosis |
|---|---|---|
| High | High | Model already knew this — well-covered topic |
| High | Low | Claim came only from retrieved context (RAG working correctly) |
| Low | High | Model may be hallucinating confidently (ignoring context) |
| Low | Low | ❌ Hallucination — neither context nor internal knowledge supports it |

---

### 5. Consistency Score

**File:** `modules/consistency.py`  
**Range:** `[0, 1]` — higher is better  
**FAST_MODE:** returns `1.0` — skips extra LLM calls  

**What it measures:**  
How stable the model's answer is across multiple independent samplings of the same question. A model that gives conflicting answers at `temperature=0.7` is less reliable than one that always says the same thing.

**How it works:**
1. Generates `NUM_CONSISTENCY_SAMPLES` (default: 3) responses at `temperature=0.7`
2. Encodes all responses with the embedding model
3. Computes average pairwise cosine similarity

$$\text{consistency} = \frac{1}{\binom{k}{2}} \sum_{i < j} \cos(\vec{v}_i, \vec{v}_j)$$

**Note:** Consistency is computed once per prompt (not per claim) and the same value is applied to all claims of that prompt.

---

### 6. FRANQ Score (Composite)

**File:** `modules/scorer.py`  
**Range:** `[0, 1]` — higher is better  
**FAST_MODE:** always runs (uses constituent scores, some of which are neutral defaults)  

**What it measures:**  
A single composite quality score for one claim, combining all the above metrics. Inspired by the FRANQ (Faithfulness, Reliability, Accuracy, NLI Quality) scoring approach.

**Formula:**
```
base = faithfulness × confidence + (1 - faithfulness) × parametric
base = base × max(0, 1 - contradiction)
franq = base × consistency
```

**Logic:**
- When faithfulness is **high**: reward grounded, confident answers (`faith × conf`)
- When faithfulness is **low**: fall back to parametric knowledge (`1-faith × parametric`)  
- Penalise contradiction regardless of faithfulness
- Scale by consistency (unstable answers are less trustworthy)

---

### 7. Claim Classification

**File:** `modules/scorer.py`  
**FAST_MODE:** always runs  

**What it is:**  
A 4-quadrant label applied to each claim based on faithfulness and parametric thresholds.

| Label | Faithfulness | Parametric | Meaning |
|---|---|---|---|
| ✅ Grounded Correct | ≥ threshold | ≥ threshold | Context supports it AND model knew it |
| ⚠️ Grounded Incorrect | ≥ threshold | < threshold | Context supports it but model didn't know |
| 🤔 Lucky Correct | < threshold | ≥ threshold | Model knew it but context doesn't confirm |
| ❌ Hallucination | < threshold | < threshold | Neither context nor internal knowledge supports it |

Thresholds: `FAITHFULNESS_THRESHOLD = 0.5`, `CONFIDENCE_THRESHOLD = 0.4` (set in `config.py`)

---

## Prompt-Level Metrics

These are computed once per prompt across all its claims or from the full response text.

---

### 8. Prompt Score

**File:** `modules/scorer.py`  
**Range:** `[0, 1]`  

Simple average of FRANQ scores across all claims in the prompt.

$$\text{prompt\_score} = \frac{1}{|C|}\sum_{c \in C}\text{franq}(c)$$

---

### 9. Grounding Score

**File:** `modules/prompt_metrics.py`  
**Range:** `[0, 1]` — higher is better  

Average faithfulness across all claims. Measures how much of the response in total is grounded in the retrieved context.

$$\text{grounding} = \frac{1}{|C|}\sum_{c \in C}\text{faithfulness}(c)$$

---

### 10. Hallucination Rate

**File:** `modules/prompt_metrics.py`  
**Range:** `[0, 1]` — lower is better  

Fraction of claims that have **both** low faithfulness AND low confidence. These are the claims most likely to be hallucinated — the model was uncertain AND the context doesn't support them.

$$\text{hallucination\_rate} = \frac{|\{c : \text{faith}(c) < \tau_f \text{ and } \text{conf}(c) < \tau_c\}|}{|C|}$$

A claim with low faithfulness but high confidence might be "confidently wrong" (different failure mode). A claim with high faithfulness but low confidence is probably fine.

---

### 11. Confidence Calibration

**File:** `modules/prompt_metrics.py`  
**Range:** `[-1, 1]`  

Pearson correlation between confidence scores and faithfulness scores across all claims in a prompt.

$$\text{calibration} = \text{corr}(\text{confidence}_c, \text{faithfulness}_c)$$

**Interpretation:**
| Value | Meaning |
|---|---|
| Close to `+1` | Well-calibrated — model is confident on grounded claims, uncertain on unsupported ones |
| Close to `0` | Confidence is uncorrelated with groundedness |
| Negative | Miscalibrated — model is more confident on unsupported claims (overconfident hallucinator) |

---

### 12. Relevance Score

**File:** `modules/prompt_metrics.py`  
**Range:** `[0, 1]` — higher is better  

Cosine similarity between the prompt embedding and the response embedding. Measures whether the response actually answers the question asked.

$$\text{relevance} = \cos(\text{embed}(\text{prompt}),\ \text{embed}(\text{response}))$$

---

### 13. Completeness Score

**File:** `modules/prompt_metrics.py`  
**Range:** `[0, 1]` — higher is better  
**Requires:** ground truth entry in `data/ground_truth.json`  

Cosine similarity between the response and the ground truth answer. Measures how much of the expected answer is covered.

$$\text{completeness} = \cos(\text{embed}(\text{response}),\ \text{embed}(\text{ground\_truth}))$$

Returns `null` when no ground truth is available for a prompt.

---

### 14. Robustness Score

**File:** `modules/prompt_metrics.py`  
**Range:** `[0, 1]` — higher is better  
**FAST_MODE:** returns `null` — skips extra LLM calls  

Measures whether the model gives consistent answers when the same question is phrased differently. A high score means the model is not brittle to phrasing changes.

**How it works:**
1. Paraphrases the prompt via Ollama (`temperature=0.3`)
2. Generates a new response to the paraphrased prompt
3. Measures cosine similarity between the original and paraphrased responses

$$\text{robustness} = \cos(\text{embed}(\text{response}_\text{original}),\ \text{embed}(\text{response}_\text{paraphrased}))$$

---

## Model-Level Metric

### 15. Model Score

**File:** `modules/scorer.py`  
**Range:** `[0, 1]`  

Average of all prompt scores — the single top-level quality number for the evaluated model.

$$\text{model\_score} = \frac{1}{|P|}\sum_{p \in P}\text{prompt\_score}(p)$$

This is the headline number shown in `outputs/results.json` under the `model_score` key.

---

## Summary Table

| Metric | Level | Range | Lower better? | Skipped in FAST_MODE? |
|---|---|---|---|---|
| Faithfulness | Claim | [0, 1] | No | No |
| Contradiction | Claim | [0, 1] | Yes | No |
| Confidence | Claim | (0, 1] | No | No |
| Parametric | Claim | [0, 1] | No | Yes (returns 0.5) |
| Consistency | Prompt | [0, 1] | No | Yes (returns 1.0) |
| FRANQ Score | Claim | [0, 1] | No | No |
| Claim Classification | Claim | Label | — | No |
| Prompt Score | Prompt | [0, 1] | No | No |
| Grounding Score | Prompt | [0, 1] | No | No |
| Hallucination Rate | Prompt | [0, 1] | Yes | No |
| Confidence Calibration | Prompt | [-1, 1] | No | No |
| Relevance Score | Prompt | [0, 1] | No | No |
| Completeness Score | Prompt | [0, 1] | No | No |
| Robustness Score | Prompt | [0, 1] | No | Yes (returns null) |
| Model Score | Global | [0, 1] | No | No |
