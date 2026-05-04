# modules/faithfulness.py — NLI-based faithfulness scoring via HuggingFace pipeline
#
# WHY pipeline instead of CrossEncoder?
# CrossEncoder.predict() returns label probabilities by *index*, which requires knowing
# the exact id2label order of the specific checkpoint. Different model versions may have
# different orderings, causing silent bugs (e.g. treating 'neutral' as 'entailment').
# pipeline(top_k=None) returns ALL labels as {"label": "entailment", "score": 0.95} so we
# match by name, which is always correct regardless of checkpoint.
#
# WHY score per document (not concatenated)?
# When 3 unrelated documents are concatenated as one premise (e.g. "computers... world
# history... internet..."), the NLI model predicts NEUTRAL because no single topic
# dominates. Scoring each document separately and taking the MAX finds the most
# supportive document, which is the semantically correct definition of faithfulness.
from transformers import pipeline as hf_pipeline
from config import NLI_MODEL

_nli = None


def _get_nli():
    """Lazy-load the NLI pipeline singleton (downloads ~500 MB on first call)."""
    global _nli
    if _nli is None:
        _nli = hf_pipeline(
            "text-classification",
            model=NLI_MODEL,
            top_k=None,        # return ALL label probabilities, not just top-1
            truncation=True,
            max_length=512,
        )
    return _nli


def _entailment_single(premise: str, hypothesis: str) -> float:
    """
    Returns the entailment probability for one (premise, hypothesis) pair.
    Matches the label by name so label-index ordering does not matter.
    """
    nli = _get_nli()
    result = nli({"text": premise, "text_pair": hypothesis})
    # result: list of {"label": str, "score": float}
    for item in result:
        if "ENTAIL" in item["label"].upper():
            return float(item["score"])
    # Fallback: LABEL_1 is entailment in most NLI model configs
    for item in result:
        if item["label"] == "LABEL_1":
            return float(item["score"])
    return 0.5  # graceful fallback: neutral


def faithfulness_score(claim: str, context_docs: list) -> float:
    """
    Returns max P(doc_i entails claim) across all retrieved context documents.

    Each document is scored independently as the NLI premise so that:
    - Irrelevant documents cannot dilute the score of the relevant one.
    - The model processes a focused, single-topic premise (< 512 tokens).

    Range: [0, 1] — higher means at least one document strongly supports the claim.
    """
    if not context_docs:
        return 0.0

    nli = _get_nli()
    # Batch all (doc, claim) pairs in a single forward pass
    inputs = [{"text": doc[:2000], "text_pair": claim} for doc in context_docs]
    batch_results = nli(inputs)
    # batch_results: list of lists — one list of label dicts per input

    best = 0.0
    for doc_result in batch_results:
        for item in doc_result:
            if "ENTAIL" in item["label"].upper():
                if item["score"] > best:
                    best = item["score"]
                break
        else:
            # LABEL_1 fallback
            for item in doc_result:
                if item["label"] == "LABEL_1" and item["score"] > best:
                    best = item["score"]

    return round(best, 4)
