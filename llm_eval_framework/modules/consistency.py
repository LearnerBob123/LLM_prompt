# modules/consistency.py — Semantic consistency via multiple re-samples
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import FAST_MODE, CONSISTENCY_MODEL, NUM_CONSISTENCY_SAMPLES, EMBEDDING_MODEL

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def consistency_score(prompt: str, context_docs: list) -> float:
    """
    Generates k responses with temperature>0, embeds them, and computes
    average pairwise cosine similarity.

    Range: [0, 1] — higher means the model gives consistent answers
    regardless of sampling randomness.

    When FAST_MODE=True returns 1.0 (skips extra LLM calls).
    """
    if FAST_MODE:
        return 1.0

    from modules.generator import generate_answer
    samples = []
    for _ in range(NUM_CONSISTENCY_SAMPLES):
        # Use higher temperature for diversity testing
        from config import OLLAMA_HOST
        import ollama
        context_str = "\n\n".join(context_docs)
        client = ollama.Client(host=OLLAMA_HOST)
        resp = client.chat(
            model=CONSISTENCY_MODEL,
            messages=[
                {"role": "system", "content": "Answer the question based on the context."},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {prompt}"},
            ],
            options={"temperature": 0.7},
        )
        samples.append(resp.message.content)

    embedder = _get_embedder()
    embeddings = embedder.encode(samples)
    n = len(embeddings)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            sims.append(sim)
    return round(float(np.mean(sims)), 4) if sims else 1.0
