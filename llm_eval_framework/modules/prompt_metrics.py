# modules/prompt_metrics.py — All prompt-level aggregated metrics
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import FAITHFULNESS_THRESHOLD, CONFIDENCE_THRESHOLD, FAST_MODE


def grounding_score(faithfulness_scores: list) -> float:
    """Average faithfulness across all claims in a prompt."""
    return round(float(np.mean(faithfulness_scores)), 4) if faithfulness_scores else 0.0


def hallucination_rate(faithfulness_scores: list, confidence_scores: list) -> float:
    """
    Fraction of claims with both low faithfulness AND low confidence.
    These are the most likely hallucinated claims.
    """
    if not faithfulness_scores:
        return 0.0
    hallucinated = sum(
        1 for f, c in zip(faithfulness_scores, confidence_scores)
        if f < FAITHFULNESS_THRESHOLD and c < CONFIDENCE_THRESHOLD
    )
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
    if FAST_MODE:
        return None

    import ollama
    from config import OLLAMA_HOST, GENERATOR_MODEL

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

    original_result = generate_fn(prompt, context_docs)
    paraphrased_result = generate_fn(paraphrased, context_docs)

    vecs = embedder.encode([original_result["response"], paraphrased_result["response"]])
    return round(float(cosine_similarity([vecs[0]], [vecs[1]])[0][0]), 4)
