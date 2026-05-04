# modules/parametric.py — Parametric knowledge scoring (context-free generation)
from config import FAST_MODE


def parametric_score(claim: str, prompt: str, nli_pipeline) -> float:
    """
    Measures whether the model's parametric (internal) knowledge supports the claim.

    Strategy:
      1. Re-generate an answer for the same prompt WITHOUT any context.
      2. Run NLI: does the context-free answer entail the claim?

    When FAST_MODE=True, returns 0.5 (neutral) to skip the extra LLM call.

    Label order for cross-encoder/nli-deberta-v3-base:
        index 0 → contradiction
        index 1 → entailment
        index 2 → neutral

    Range: [0, 1]
    """
    if FAST_MODE:
        return 0.5  # neutral default — neither penalises nor rewards

    from modules.generator import generate_answer
    result = generate_answer(prompt, context_docs=[], with_context=False)
    context_free_response = result["response"]

    scores = nli_pipeline.predict(
        [(context_free_response, claim)], apply_softmax=True
    )
    entailment_prob = float(scores[0][1])
    return round(entailment_prob, 4)
