from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever


def build_bm25(config, documents):
    bm25_config = config["bm25"]

    return BM25Retriever(
        documents=documents,
        k1=bm25_config.get("k1", 1.5),
        b=bm25_config.get("b", 0.75),
        epsilon=bm25_config.get("epsilon", 0.25),
    )


def build_dense(config, documents):
    dense_config = config["dense"]

    retriever = DenseRetriever(
        model_name=dense_config["model_name"],
        batch_size=dense_config.get("batch_size", 64),
        device=dense_config.get("device"),
    )

    retriever.index_documents(documents)
    return retriever


def build_dense_finetuned(config, documents):
    dense_config = config["dense_finetuned"]

    retriever = DenseRetriever(
        model_name=dense_config["model_name"],
        batch_size=dense_config.get("batch_size", 64),
        device=dense_config.get("device"),
    )

    retriever.index_documents(documents)
    return retriever


RETRIEVER_BUILDERS = {
    "bm25": build_bm25,
    "dense": build_dense,
    "dense_finetuned": build_dense_finetuned,
}


def build_retriever(model_name, config, documents):
    if model_name not in RETRIEVER_BUILDERS:
        raise ValueError(f"Unknown retriever: {model_name}")

    return RETRIEVER_BUILDERS[model_name](config, documents)