# modules/confidence.py — Log-probability confidence scoring
import math


def confidence_score(token_logprobs: list) -> float:
    """
    Computes normalized log-likelihood (inverse perplexity proxy).
    Formula: exp( mean( log P(token_t) ) )

    Range: (0, 1] — higher means the model generated the response with higher
    average token probability, i.e. it was more "confident".

    Returns 0.0 when no logprobs are available (Ollama may not return them for
    all model variants).
    """
    if not token_logprobs:
        return 0.0
    # Guard against +inf / -inf values occasionally returned by some models
    valid = [lp for lp in token_logprobs if math.isfinite(lp)]
    if not valid:
        return 0.0
    avg_logprob = sum(valid) / len(valid)
    # Clamp to avoid exp overflow / underflow
    avg_logprob = max(-20.0, min(0.0, avg_logprob))
    return round(math.exp(avg_logprob), 4)
