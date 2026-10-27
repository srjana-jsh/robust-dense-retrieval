from __future__ import annotations

from dataclasses import dataclass

import ir_datasets


@dataclass
class RetrievalDataset:
    documents: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def load_retrieval_dataset(
    dataset_name: str,
) -> RetrievalDataset:
    """
    Load a retrieval dataset from ir_datasets.

    Returns:
        documents:
            doc_id -> document text

        queries:
            query_id -> query text

        qrels:
            query_id -> {doc_id: relevance}
    """

    dataset = ir_datasets.load(dataset_name)

    documents = {
        doc.doc_id: _document_text(doc)
        for doc in dataset.docs_iter()
    }

    queries = {
        query.query_id: query.text
        for query in dataset.queries_iter()
    }

    qrels: dict[str, dict[str, int]] = {}

    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, {})
        qrels[qrel.query_id][qrel.doc_id] = int(qrel.relevance)

    return RetrievalDataset(
        documents=documents,
        queries=queries,
        qrels=qrels,
    )


def _document_text(doc) -> str:
    """
    Build one text field from a document object.

    Some datasets have title + text, while others have
    only text. Empty text is allowed.
    """
    parts = []

    title = getattr(doc, "title", None)
    if title is not None:
        parts.append(str(title))

    text = getattr(doc, "text", None)
    if text is not None:
        parts.append(str(text))

    body = getattr(doc, "body", None)
    if body is not None:
        parts.append(str(body))

    if not parts:
        raise ValueError(
            f"Could not find a usable text field for document type {type(doc).__name__}."
        )

    return " ".join(parts).strip()