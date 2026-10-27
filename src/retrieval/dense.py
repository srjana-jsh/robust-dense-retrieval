from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """
    Pretrained bi-encoder retriever using cosine similarity.

    Documents are encoded once and stored in a FAISS index.
    Queries are encoded with the same model and searched against
    the document embeddings.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.doc_ids: list[str] = []
        self.index = None

    def index_documents(self, documents: dict[str, str]) -> None:
        self.doc_ids = list(documents.keys())
        document_texts = [documents[doc_id] for doc_id in self.doc_ids]

        embeddings = self.model.encode(
            document_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype="float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve_all(
        self,
        queries: dict[str, str],
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        if self.index is None:
            raise RuntimeError("Call index_documents() before retrieval.")

        query_ids = list(queries.keys())
        query_texts = [queries[query_id] for query_id in query_ids]

        query_embeddings = self.model.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embeddings = np.asarray(query_embeddings, dtype="float32")

        top_k = min(top_k, len(self.doc_ids))

        if top_k == 0:
            return {query_id: [] for query_id in query_ids}

        scores, indices = self.index.search(query_embeddings, top_k)

        results = {}

        for row, query_id in enumerate(query_ids):
            results[query_id] = [
                (self.doc_ids[doc_index], float(score))
                for doc_index, score in zip(indices[row], scores[row])
                if doc_index >= 0
            ]

        return results