# modules/retriever.py — FAISS-based document retrieval over the knowledge base
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import config
from config import EMBEDDING_MODEL, TOP_K_DOCS


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.documents = []
        self.index = None
        self._build_index()

    def _build_index(self):
        kb_path = config.KNOWLEDGE_BASE_DIR
        # Support running from either the project root or the framework subfolder
        if not os.path.exists(kb_path):
            kb_path = os.path.join(os.path.dirname(__file__), "..", config.KNOWLEDGE_BASE_DIR)

        for fname in sorted(os.listdir(kb_path)):
            if fname.endswith(".txt"):
                with open(os.path.join(kb_path, fname), encoding="utf-8") as f:
                    self.documents.append(f.read().strip())

        if not self.documents:
            raise RuntimeError(f"No .txt files found in {config.KNOWLEDGE_BASE_DIR}")

        embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = TOP_K_DOCS) -> list:
        query_vec = self.model.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(query_vec, top_k)
        return [self.documents[i] for i in indices[0]]
