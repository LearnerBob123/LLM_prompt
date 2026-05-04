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
#
# WHY sentence-level extraction before NLI?
# NLI models like nli-deberta-v3-base are trained on sentence-level premise/hypothesis
# pairs (SNLI, MultiNLI). When given a 2000-character multi-paragraph document as the
# premise, the model sees a broad topic overview and predicts NEUTRAL — even if the
# claim appears word-for-word in the text. Extracting the 3 most relevant sentences
# (by cosine similarity to the claim) gives the NLI model a focused, sentence-level
# premise it was actually trained to handle, producing correct entailment scores.
import re
import numpy as np
from transformers import pipeline as hf_pipeline
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sentence_transformers import SentenceTransformer
from config import NLI_MODEL, EMBEDDING_MODEL

NLI_TOP_SENTENCES = 3   # number of most-relevant sentences used as NLI premise

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_sentence_candidates(doc: str, claim: str, top_k: int = NLI_TOP_SENTENCES) -> list:
    """
    Returns up to top_k individual sentences from doc most similar to the claim.
    Each sentence has whitespace normalized (newlines converted to spaces).
    NaN cosine scores (from zero-norm embeddings) are replaced with -inf so they
    are never selected as candidates.

    Returns a list[str] — NOT a joined string — so callers can score each sentence
    individually as an NLI premise. Scoring per-sentence avoids confusing the
    cross-encoder with multi-sentence premises that mix unrelated topics.
    """
    sentences = [re.sub(r'\s+', ' ', s.strip())
                 for s in re.split(r'(?<=[.!?])\s+', doc)
                 if len(s.strip()) > 20]
    if not sentences:
        return [doc[:500].replace('\n', ' ')]
    if len(sentences) <= top_k:
        return sentences
    embedder = _get_embedder()
    sent_vecs = embedder.encode(sentences)
    claim_vec = embedder.encode([claim])
    scores = cos_sim(claim_vec, sent_vecs)[0]
    # Replace NaN (zero-norm vector edge case) with -inf so they are never top-ranked
    scores = np.where(np.isnan(scores), -np.inf, scores)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [sentences[i] for i in top_indices]


def _extract_relevant_sentences(doc: str, claim: str, top_k: int = NLI_TOP_SENTENCES) -> str:
    """Kept for backward compatibility. Returns top_k sentences joined as a string."""
    return ' '.join(_get_sentence_candidates(doc, claim, top_k))

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


def faithfulness_and_contradiction(claim: str, context_docs: list) -> tuple:
    """
    Computes faithfulness and contradiction in a single NLI pass (half the inference cost).

    Key design decision: contradiction is reported from the sentence that gave the BEST
    ENTAILMENT score, not from any arbitrary sentence. This prevents a well-known false-
    positive pattern:
      - Sentence S1: "ENIAC was completed at the University of Pennsylvania."
        → entailment=0.998, contradiction=0.001
      - Sentence S2 (topically related but different claim): "neither was fully built in
        his lifetime."
        → entailment=0.02, contradiction=0.95

    Without this fix, max(contradiction) = 0.95 which destroys the FRANQ score for a
    perfectly grounded claim. By anchoring contradiction to the best-entailment sentence,
    we get faith=0.998, contra=0.001, which correctly reflects context support.

    This also covers genuine hallucinations: if the best-entailment sentence has
    entailment=0.05 (no support) and contradiction=0.93, those values are reported
    faithfully, correctly penalizing the claim in the FRANQ formula.

    Returns:
        (faithfulness, contradiction) — both floats in [0, 1]
    """
    if not context_docs:
        return 0.0, 0.0

    nli = _get_nli()
    best_entail = 0.0
    contra_of_best = 0.0   # contradiction score of the best-entailment sentence

    for doc in context_docs:
        candidates = _get_sentence_candidates(doc, claim)
        inputs = [{"text": s, "text_pair": claim} for s in candidates]
        batch_results = nli(inputs)
        # batch_results: list of lists — one per candidate sentence

        for sent_result in batch_results:
            entail = 0.0
            contra = 0.0
            for item in sent_result:
                lbl = item["label"].upper()
                if "ENTAIL" in lbl:
                    entail = item["score"]
                elif "CONTRADICT" in lbl:
                    contra = item["score"]
                elif lbl == "LABEL_1":   # entailment fallback for unnamed models
                    entail = item["score"]
                elif lbl == "LABEL_0":   # contradiction fallback for unnamed models
                    contra = item["score"]

            if entail > best_entail:
                best_entail = entail
                contra_of_best = contra   # track contradiction of the best-support sentence

    return round(best_entail, 4), round(contra_of_best, 4)


def faithfulness_score(claim: str, context_docs: list) -> float:
    """
    Returns max P(sentence_j entails claim) across all candidate sentences in all docs.
    Prefer faithfulness_and_contradiction() when both scores are needed (single NLI pass).

    Range: [0, 1] — higher means at least one context sentence strongly supports the claim.
    """
    faith, _ = faithfulness_and_contradiction(claim, context_docs)
    return faith
