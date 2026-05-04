# modules/contradiction.py — NLI-based contradiction detection via CrossEncoder
# Shares the NLI singleton loaded in faithfulness.py
from modules.faithfulness import _get_nli


def contradiction_score(claim: str, context_docs: list) -> float:
    """
    Returns P(claim contradicts context).

    Label order for cross-encoder/nli-deberta-v3-base:
        index 0 → contradiction
        index 1 → entailment
        index 2 → neutral

    Range: [0, 1] — higher means the claim actively contradicts the context.
    """
    nli = _get_nli()
    context = " ".join(context_docs)[:1500]
    scores = nli.predict([(context, claim)], apply_softmax=True)
    contradiction_prob = float(scores[0][0])
    return round(contradiction_prob, 4)
