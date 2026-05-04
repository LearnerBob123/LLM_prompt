# modules/scorer.py — FRANQ-style composite scoring and error taxonomy
from config import FAITHFULNESS_THRESHOLD, CONFIDENCE_THRESHOLD


def franq_score(
    faithfulness: float,
    confidence: float,
    parametric: float,
    contradiction: float,
    consistency: float,
) -> float:
    """
    FRANQ-style composite claim score.

    Logic:
      - When faithfulness is high  → reward grounded answer (faith * conf)
      - When faithfulness is low   → fall back to parametric knowledge
      - Penalise contradiction
      - Scale by consistency (how stable are outputs to prompt rephrasing)

    Range: [0, 1]
    """
    base = faithfulness * confidence + (1 - faithfulness) * parametric
    base *= max(0.0, 1.0 - contradiction)
    base *= consistency
    return round(max(0.0, min(1.0, base)), 4)


def classify_claim(faithfulness: float, parametric: float) -> str:
    """
    4-quadrant error taxonomy.

    Faithfulness  | Parametric  | Label
    --------------|-------------|-------------------------------
    High          | High        | ✅ Grounded Correct
    High          | Low         | ⚠️ Grounded Incorrect
    Low           | High        | 🤔 Lucky Correct
    Low           | Low         | ❌ Hallucination
    """
    is_faithful = faithfulness >= FAITHFULNESS_THRESHOLD
    is_factual = parametric >= CONFIDENCE_THRESHOLD

    if is_faithful and is_factual:
        return "✅ Grounded Correct"
    elif is_faithful and not is_factual:
        return "⚠️ Grounded Incorrect"
    elif not is_faithful and is_factual:
        return "🤔 Lucky Correct"
    else:
        return "❌ Hallucination"


def prompt_score(claim_scores: list) -> float:
    """Average FRANQ score across all claims in a single prompt."""
    return round(sum(claim_scores) / len(claim_scores), 4) if claim_scores else 0.0


def model_score(prompt_scores: list) -> float:
    """Average prompt score across all prompts — the overall model quality score."""
    return round(sum(prompt_scores) / len(prompt_scores), 4) if prompt_scores else 0.0
