from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    def __init__(
        self,
        documents: dict[str, str],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.doc_ids = list(documents.keys())

        tokenized_corpus = [
            tokenize(documents[doc_id])
            for doc_id in self.doc_ids
        ]

        self.index = BM25Okapi(
            tokenized_corpus,
            k1=k1,
            b=b,
            epsilon=epsilon,
        )

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        scores = self.index.get_scores(tokenize(query))
        top_k = min(top_k, len(self.doc_ids))

        if top_k == 0:
            return []

        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        ranked_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        return [
            (self.doc_ids[index], float(scores[index]))
            for index in ranked_indices
        ]

    def retrieve_all(
        self,
        queries: dict[str, str],
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        results = {}

        for query_id, query in tqdm(queries.items(), desc="Running BM25"):
            results[query_id] = self.search(query, top_k)

        return results