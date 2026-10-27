from __future__ import annotations

import math


def evaluate_retrieval(
    results: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    """Compute retrieval metrics averaged across judged queries."""

    ndcg_10 = []
    recall_10 = []
    recall_100 = []
    mrr_10 = []

    for query_id, relevance in qrels.items():
        if query_id not in results:
            continue

        ranked_doc_ids = [doc_id for doc_id, _ in results[query_id]]

        ndcg_10.append(_ndcg_at_k(ranked_doc_ids, relevance, k=10))
        recall_10.append(_recall_at_k(ranked_doc_ids, relevance, k=10))
        recall_100.append(_recall_at_k(ranked_doc_ids, relevance, k=100))
        mrr_10.append(_mrr_at_k(ranked_doc_ids, relevance, k=10))

    if not ndcg_10:
        raise ValueError("No evaluated queries were found.")

    return {
        "ndcg@10": _mean(ndcg_10),
        "recall@10": _mean(recall_10),
        "recall@100": _mean(recall_100),
        "mrr@10": _mean(mrr_10),
    }


def _ndcg_at_k(
    ranked_doc_ids: list[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    dcg = 0.0

    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = relevance.get(doc_id, 0)
        gain = (2**rel) - 1
        dcg += gain / math.log2(rank + 1)

    ideal_relevances = sorted(relevance.values(), reverse=True)[:k]

    idcg = 0.0

    for rank, rel in enumerate(ideal_relevances, start=1):
        gain = (2**rel) - 1
        idcg += gain / math.log2(rank + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def _recall_at_k(
    ranked_doc_ids: list[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    relevant_docs = {
        doc_id
        for doc_id, rel in relevance.items()
        if rel > 0
    }

    if not relevant_docs:
        return 0.0

    retrieved = set(ranked_doc_ids[:k])

    return len(relevant_docs.intersection(retrieved)) / len(relevant_docs)


def _mrr_at_k(
    ranked_doc_ids: list[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if relevance.get(doc_id, 0) > 0:
            return 1.0 / rank

    return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)