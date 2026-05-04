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
    from modules.faithfulness import _entailment_single
    return round(_entailment_single(context_free_response, claim), 4)
