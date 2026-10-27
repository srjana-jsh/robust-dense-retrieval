# Robust Dense Retrieval Under Domain Shift

A small research project for studying how retrieval systems behave when moved to a new domain.

## Current scope

The repository currently implements:

1. Dataset loading and basic EDA.
2. BM25 lexical retrieval baseline.
3. Pretrained dense bi-encoder baseline.

Later stages can add hard-negative mining, target-domain adaptation, pseudo-labeling, hybrid retrieval, and distributed training.

## Research question

> How robust are retrieval systems under domain shift, and which adaptation strategies recover retrieval quality most efficiently?

## Dataset

The default configuration uses:

```text
beir/scifact/test
```

Other useful target domains include:

```text
beir/fiqa/test
beir/nfcorpus/test
```

Change the dataset in `configs/base.yaml`.

## Configuration

`configs/base.yaml` contains the settings shared by the baselines:

```yaml
dataset:
  name: beir/scifact/test

retrieval:
  top_k: 100

dense:
  model_name: sentence-transformers/msmarco-MiniLM-L6-cos-v5
  batch_size: 64
  device: null
```

`device: null` lets Sentence Transformers choose CUDA when available and CPU otherwise.

## Install

```bash
pip install -r requirements.txt
```

## Data exploration

```bash
jupyter notebook notebooks/01_data_eda.ipynb
```

## BM25 baseline

```bash
python scripts/evaluate_bm25.py
```

## Dense bi-encoder baseline

```bash
python scripts/evaluate_dense.py
```

The dense baseline uses a pretrained MS MARCO bi-encoder. Documents are encoded once, L2-normalized, and stored in a FAISS inner-product index. Because query embeddings are also normalized, inner product corresponds to cosine similarity.

## Metrics

Both baselines use the same evaluation code and report:

- nDCG@10
- Recall@10
- Recall@100
- MRR@10

Results are saved in `reports/results/`.
