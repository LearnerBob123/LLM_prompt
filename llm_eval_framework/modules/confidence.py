# modules/confidence.py — Log-probability confidence scoring
#
# Two functions:
#   confidence_score(token_logprobs)         — whole-response confidence
#   claim_confidence_score(claim, token_data) — per-claim confidence via token alignment
#
# HOW PER-CLAIM CONFIDENCE WORKS:
# Ollama returns per-token logprobs WITH the decoded token strings (lp.token).
# We concatenate those strings to reconstruct the full response text, then locate
# the claim substring within it. The logprobs of the matching tokens become the
# claim's confidence score. This is much more meaningful than assigning every claim
# in a response the same score.
import math


def confidence_score(token_logprobs: list) -> float:
    """
    Whole-response confidence: exp( mean( log P(token_t) ) )

    This is the geometric mean of per-token probabilities, equivalent to
    inverse perplexity^(1/N). Range: (0, 1] — higher = more confident.

    Returns 0.0 when no logprobs are available (some Ollama model variants
    do not return them).
    """
    if not token_logprobs:
        return 0.0
    valid = [lp for lp in token_logprobs if math.isfinite(lp)]
    if not valid:
        return 0.0
    avg_logprob = sum(valid) / len(valid)
    avg_logprob = max(-20.0, min(0.0, avg_logprob))   # clamp to avoid overflow
    return round(math.exp(avg_logprob), 4)


def claim_confidence_score(claim_text: str, token_data: list) -> float:
    """
    Per-claim confidence score computed by aligning the claim text to the
    response token sequence and averaging only those tokens' log-probabilities.

    token_data: list of {"token": str, "logprob": float} dicts
                (produced by generator.py when logprobs=True)

    HOW THE ALIGNMENT WORKS:
    1. Concatenate token strings to reconstruct the full response text.
    2. Find the start index of the claim substring (tries progressively shorter
       prefixes if the full claim isn't found verbatim).
    3. Find all tokens whose character span overlaps [claim_start, claim_end].
    4. Compute confidence_score() on only those token logprobs.

    Falls back to whole-response confidence_score() if alignment fails
    (e.g. the claim was heavily paraphrased during extraction).
    """
    if not token_data:
        return 0.0

    tokens = [td["token"] for td in token_data]
    logprobs = [td["logprob"] for td in token_data]

    # Build the full response text and record each token's start character position
    start_positions = []
    running = ""
    for tok in tokens:
        start_positions.append(len(running))
        running += tok
    full_text = running

    # --- Find claim span via substring search ---
    claim_stripped = claim_text.strip()
    search_text = full_text.lower()
    search_claim = claim_stripped.lower()

    # Try increasingly short prefixes until we find a match
    idx = -1
    for prefix_len in [len(search_claim), 60, 40, 20]:
        if prefix_len > len(search_claim):
            continue
        pos = search_text.find(search_claim[:prefix_len])
        if pos != -1:
            idx = pos
            break

    if idx == -1:
        # Alignment failed — fall back to whole-response confidence
        return confidence_score(logprobs)

    claim_end = idx + len(claim_stripped)

    # --- Find which tokens overlap the claim span ---
    claim_indices = []
    for i, (start, tok) in enumerate(zip(start_positions, tokens)):
        tok_end = start + len(tok)
        if tok_end > idx and start < claim_end:
            claim_indices.append(i)

    if not claim_indices:
        return confidence_score(logprobs)

    claim_logprobs = [logprobs[i] for i in claim_indices]
    return confidence_score(claim_logprobs)
