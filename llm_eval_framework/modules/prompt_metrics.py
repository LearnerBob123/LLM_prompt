# modules/prompt_metrics.py — All prompt-level aggregated metrics
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import config   # import the MODULE so config.FAST_MODE reflects runtime changes
from config import FAITHFULNESS_THRESHOLD, CONFIDENCE_THRESHOLD


def grounding_score(faithfulness_scores: list) -> float:
    """Average faithfulness across all claims in a prompt."""
    return round(float(np.mean(faithfulness_scores)), 4) if faithfulness_scores else 0.0


def hallucination_rate(faithfulness_scores: list, confidence_scores: list) -> float:
    """
    Fraction of claims that are NOT supported by the retrieved context.

    A claim is counted as hallucinated when faithfulness < FAITHFULNESS_THRESHOLD,
    regardless of the model's confidence. This captures the most dangerous failure
    mode: an LLM that is highly confident but factually wrong (overconfident
    hallucination). The previous definition (faith low AND conf low) missed these
    because LLM token-logprob confidence is almost always > 0.5.

    Range: [0, 1] — 0.0 = all claims grounded; 1.0 = all claims hallucinated.
    """
    if not faithfulness_scores:
        return 0.0
    hallucinated = sum(1 for f in faithfulness_scores if f < FAITHFULNESS_THRESHOLD)
    return round(hallucinated / len(faithfulness_scores), 4)


def confidence_calibration(confidence_scores: list, faithfulness_scores: list) -> float:
    """
    Pearson correlation between confidence and faithfulness.
    High positive correlation = model is well-calibrated (confident on faithful claims).
    Range: [-1, 1]
    """
    if len(confidence_scores) < 2:
        return 0.0
    result = np.corrcoef(confidence_scores, faithfulness_scores)[0, 1]
    if np.isnan(result):
        return 0.0
    return round(float(result), 4)


def relevance_score(prompt: str, response: str, embedder) -> float:
    """Cosine similarity between prompt and response embeddings."""
    vecs = embedder.encode([prompt, response])
    return round(float(cosine_similarity([vecs[0]], [vecs[1]])[0][0]), 4)


def completeness_score(response: str, ground_truth: str, embedder) -> float:
    """
    Semantic similarity of response to ground truth.
    Returns None when no ground truth is available.
    """
    if not ground_truth:
        return None
    vecs = embedder.encode([response, ground_truth])
    return round(float(cosine_similarity([vecs[0]], [vecs[1]])[0][0]), 4)


def robustness_score(prompt: str, context_docs: list, generate_fn, embedder) -> float:
    """
    Paraphrases the prompt via Ollama, re-generates, then measures cosine similarity
    between the original and paraphrased responses.

    When FAST_MODE=True returns None (skips extra LLM calls).
    """
    if config.FAST_MODE:
        return None

    import time
    import ollama
    from config import OLLAMA_HOST, GENERATOR_MODEL

    # Retry paraphrase call — stale connections can occur after long NLI GPU runs
    last_err = None
    for attempt in range(3):
        try:
            client = ollama.Client(host=OLLAMA_HOST)
            paraphrase_result = client.chat(
                model=GENERATOR_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"Rewrite this question differently while keeping the same meaning. "
                               f"Output only the rewritten question, nothing else.\n\n{prompt}"
                }],
                options={"temperature": 0.3},
            )
            paraphrased = paraphrase_result.message.content.strip()
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)
    else:
        print(f"[robustness] Ollama error after 3 attempts: {last_err}")
        return None

    original_result = generate_fn(prompt, context_docs)
    paraphrased_result = generate_fn(paraphrased, context_docs)

    vecs = embedder.encode([original_result["response"], paraphrased_result["response"]])
    return round(float(cosine_similarity([vecs[0]], [vecs[1]])[0][0]), 4)
