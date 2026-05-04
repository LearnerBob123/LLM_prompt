# modules/contradiction.py — NLI-based contradiction detection
#
# Shares the NLI pipeline singleton from faithfulness.py so the model is only
# loaded into memory once. Also re-uses _extract_relevant_sentences so contradiction
# scoring is focused on the most topically relevant sentences (same reason as faithfulness).
from modules.faithfulness import faithfulness_and_contradiction


def contradiction_score(claim: str, context_docs: list) -> float:
    """
    Returns the contradiction score for the most-supported sentence.

    Internally calls faithfulness_and_contradiction() and returns the contradiction
    component. Contradiction is anchored to the sentence with the highest entailment
    score, preventing false-positive contradiction from off-topic candidate sentences
    that happen to be topically similar but semantically unrelated to the specific claim.

    Interpretation guide:
      - Low faithfulness + LOW  contradiction → context is silent on this claim
      - Low faithfulness + HIGH contradiction → claim conflicts with what the context says
      - High faithfulness + LOW contradiction → claim well-supported (✅ Grounded Correct)

    Range: [0, 1]
    """
    _, contra = faithfulness_and_contradiction(claim, context_docs)
    return contra
