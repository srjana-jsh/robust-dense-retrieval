from __future__ import annotations

import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.datasets import load_retrieval_dataset
from evaluation.metrics import evaluate_retrieval
from retrieval.factory import build_retriever
from utils.results import print_metrics, save_results


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(models):
    config = load_config(PROJECT_ROOT / "configs/base.yaml")
    top_k = config["retrieval"]["top_k"]

    model_results = {model_name: {} for model_name in models}

    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        dataset_label = dataset_config["label"]
        max_documents = dataset_config.get("max_documents")
        seed = dataset_config.get("seed", 42)

        print(f"\nLoading dataset: {dataset_name}")

        dataset = load_retrieval_dataset(
            dataset_name=dataset_name,
            max_documents=max_documents,
            seed=seed,
        )

        print(f"Documents: {len(dataset.documents):,}")
        print(f"Queries:   {len(dataset.queries):,}")

        for model_name in models:
            print(f"\nRunning model: {model_name}")

            retriever = build_retriever(
                model_name=model_name,
                config=config,
                documents=dataset.documents,
            )

            results = retriever.retrieve_all(
                queries=dataset.queries,
                top_k=top_k,
            )

            metrics = evaluate_retrieval(results, dataset.qrels)

            print_metrics(
                title=f"{model_name} - {dataset_label}",
                metrics=metrics,
            )

            model_results[model_name][dataset_label] = {
                "dataset": dataset_name,
                "documents": len(dataset.documents),
                "queries": len(dataset.queries),
                "sampled": max_documents is not None,
                "top_k": top_k,
                "metrics": metrics,
            }

    for model_name in models:
        output_path = save_results(
            project_root=PROJECT_ROOT,
            method=model_name,
            results=model_results[model_name],
            settings=config.get(model_name),
        )

        print(f"\nSaved {model_name} results to: {output_path}")


if __name__ == "__main__":
    main(
        models=[
            "bm25",
            "dense",
        ]
    )