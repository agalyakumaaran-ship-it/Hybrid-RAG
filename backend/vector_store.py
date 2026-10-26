
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# A small, multilingual-capable embedding model.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.chunks = []

    def build(self, chunks: list[str]) -> None:
        if not chunks:
            raise ValueError("No document chunks provided.")

        self.chunks = chunks

        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        self.index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )
        self.index.add(embeddings)

    def search(self, question: str, top_k: int = 3) -> list[str]:
        if self.index is None:
            raise ValueError("Build the vector index first.")

        question_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        question_embedding = np.asarray(
            question_embedding,
            dtype="float32",
        )

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(
            question_embedding,
            k,
        )

        return [
            self.chunks[i]
            for i in indices[0]
            if i >= 0
        ]
