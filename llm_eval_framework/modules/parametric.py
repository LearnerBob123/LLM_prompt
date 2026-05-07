# modules/parametric.py — Parametric knowledge scoring (context-free generation)
#
# Measures whether the model's *internal* knowledge (no retrieved context) supports
# the claim. Uses _entailment_single() from faithfulness.py — same NLI pipeline,
# same label-name matching, no CrossEncoder dependency.


def parametric_score(claim: str, context_free_response: str) -> float:
    """
    Measures whether the model's parametric (internal) knowledge supports the claim.

    Args:
        claim: the atomic claim to evaluate
        context_free_response: a pre-generated answer produced WITHOUT retrieved context
                               (generate once per prompt, reuse for all claims)

    Strategy:
      Run NLI: does the context-free answer entail the claim?

    Interpretation:
      High faithfulness + HIGH parametric → model already knew the fact (well-covered topic)
      High faithfulness + LOW  parametric → claim came only from the retrieved context (RAG working)
      Low  faithfulness + HIGH parametric → model may be hallucinating confidently

    Range: [0, 1]
    """
    import config
    if config.FAST_MODE:
        return 0.5  # neutral — skipped in fast mode

    # Extract the most relevant sentences from the context-free response before NLI,
    # exactly as faithfulness.py does for context documents. Passing a full paragraph
    # as the NLI premise causes the model to predict NEUTRAL (it's trained on sentence
    # pairs), which gives near-0 entailment for all claims — the bug this fixes.
    from modules.faithfulness import _entailment_single, _get_sentence_candidates
    candidates = _get_sentence_candidates(context_free_response, claim)
    premise = ' '.join(candidates)
    return round(_entailment_single(premise, claim), 4)
