# PoliticHeadlinES TFG

Ranking models for the **PoliticHeadlinES-IberLEF 2026** shared task. The task consists of ranking ten candidate headlines for each Spanish political news article. The project explores lexical baselines, semantic models, fine-tuned cross-encoders, weighted textual ensembles, reranking strategies, LLM-based ranking and multimodal fusion with vision-language models.

This repository contains the code developed for the Bachelor's Thesis **"Language Models for the Automatic Detection and Ranking of Headlines in Political News"**.

## Recent Update: Refactoring and Documentation

This repository has been refactored and documented to improve readability, maintainability and reproducibility. The latest changes include:

- Centralized repeated ranking, scoring, inference and model-building utilities into shared modules.
- Simplified the main execution pipeline and the experimental evaluation script.
- Removed obsolete training wrappers and consolidated evaluation functionality into a single `evaluate_experiments.py` script.
- Translated logs, comments, docstrings and error messages to English.
- Added clearer documentation for baselines, cross-encoders, rerankers, LLM experiments and VLM fusion.

### Planned Improvements

Future improvements may include cleaning remaining compatibility or obsolete code, and adding a dedicated Colab notebook to load and run the project more easily.

## Overview

Given an article, ten candidate headlines and, optionally, an associated image, the system returns two rankings:

- **Task 1:** text-only headline ranking.
- **Task 2:** multimodal ranking, optionally combining textual relevance with image-text similarity.

The main local evaluation metric is **PA-nDCG@10**, the official task metric. The code also reports top-1 accuracy for local analysis.

## Repository structure

```text
.
├── evaluate_experiments.py        # Batch evaluation script for experimental runs
├── split_dataset.py               # Train/validation/test split generation
├── dataset_metrics.py             # Dataset statistics and split inspection
├── analyze_ranking_errors.py      # Error analysis utilities
├── requirements.txt               # Python dependencies
├── src/
│   ├── config.py                  # Global paths, model selection and hyperparameters
│   ├── run.py                     # Main inference and local evaluation pipeline
│   ├── models/
│   │   ├── factory.py             # Centralized ranker construction
│   │   ├── modeling_utils.py      # Shared Transformer/model utilities
│   │   ├── tfidf_ranker.py        # TF-IDF baseline
│   │   ├── bm25_ranker.py         # BM25 baseline
│   │   ├── semantic_ranker.py     # Sentence embedding baseline
│   │   ├── cross_encoder_ranker.py
│   │   ├── cross_encoder_rank10_ranker.py
│   │   ├── cross_encoder_ensemble_ranker.py
│   │   ├── tail_reranker.py
│   │   ├── modern_reranker.py     # BGE reranker pipeline
│   │   ├── llm_ranker.py          # LLM-based ranking experiments
│   │   └── vlm_ranker.py          # CLIP/SigLIP multimodal fusion
│   ├── training/
│   │   ├── train_crossencoder.py
│   │   └── train_crossencoder_rank10.py
│   └── utils/
│       ├── data_utils.py
│       ├── data_pairs.py
│       ├── inference.py
│       ├── metrics.py
│       ├── reproducibility.py
│       ├── scoring.py
│       └── submission.py
```

## Data and checkpoints

The dataset and trained checkpoints are not included in this repository because they can be large and are subject to external distribution rules.

By default, `src/config.py` expects the following structure:

```text
data/
└── tfg_split/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── images/
```

The expected CSV columns are:

```text
id, article_body, image_hash, title_1, ..., title_10
```

For local evaluation and training, a `y_true` column is also required. It should contain the gold ranking using tokens such as:

```text
t3 t1 t5 t2 t4 t6 t7 t8 t9 t10
```

Trained model checkpoints are expected under `outputs/`, for example:

```text
outputs/
├── beto_model/
├── beto_headtail_model/
├── bertin_model/
├── mdeberta_model/
└── beto_rank10_model/
```

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

A GPU is recommended for cross-encoders, LLMs and VLMs. Lexical baselines such as TF-IDF and BM25 can run on CPU.

## Configuration

Most options are defined in `src/config.py`:

- active dataset paths;
- selected model for `src/run.py`;
- cross-encoder hyperparameters;
- ensemble weights;
- reranking settings;
- LLM and VLM configuration;
- output paths.

The main model is selected with:

```python
MODEL_NAME = "beto"
```

Supported values include:

```text
tfidf, bm25, semantic, beto, beto_headtail, bertin, mdeberta,
crossencoder_ensemble, beto_rank10, tail_reranker, modern_reranker,
llm_ranker
```

Multimodal fusion for Task 2 can be enabled with:

```python
USE_VLM_FOR_TASK2 = True
VLM_BACKEND = "siglip"  # or "clip"
```

## Running the main pipeline

From the repository root:

```bash
python src/run.py
```

This will:

1. load the configured train/test files;
2. build the selected ranker;
3. generate Task 1 rankings;
4. optionally generate Task 2 rankings using VLM fusion;
5. save the submission file under `outputs/`;
6. compute local metrics when `y_true` is available.

For a light smoke test, use a non-trained baseline in `src/config.py`:

```python
MODEL_NAME = "bm25"
USE_VLM_FOR_TASK2 = False
```

## Training models

Train a pointwise binary cross-encoder:

```bash
python src/training/train_crossencoder.py beto
python src/training/train_crossencoder.py beto_headtail
python src/training/train_crossencoder.py bertin
python src/training/train_crossencoder.py mdeberta
```

Train the rank10 regression-based cross-encoder:

```bash
python src/training/train_crossencoder_rank10.py beto_rank10
```

Training configurations, checkpoints and training history are saved under the corresponding directory in `outputs/`.

## Experimental evaluation

`evaluate_experiments.py` is used to reproduce and compare multiple experimental stages.

Evaluate lightweight individual baselines:

```bash
python evaluate_experiments.py --stage individual --models tfidf,bm25 --limit 5
```

Evaluate the final textual ensemble:

```bash
python evaluate_experiments.py --stage ensembles \
  --ensemble-specs "beto_headtail:0.40,beto:0.45,mdeberta:0.15" \
  --limit 5
```

Evaluate VLM fusion using the tail-reranked textual base:

```bash
python evaluate_experiments.py --stage vlm \
  --base-ensemble "beto_headtail:0.40,beto:0.45,mdeberta:0.15" \
  --vlm-text-base tail_rank10 \
  --limit 5
```

Available stages include:

```text
individual, ensembles, rerankers, llm, vlm, all_text
```

Evaluation outputs are saved under:

```text
outputs/evaluation/
```

## Main modeling approaches

| Family | Description |
| --- | --- |
| TF-IDF | Lexical baseline using cosine similarity between article and headlines. |
| BM25 | Lexical ranking baseline with configurable `k1`, `b` and query length. |
| Semantic embeddings | Sentence embedding baseline with cosine similarity. |
| Cross-encoders | Fine-tuned Transformer models scoring each article-headline pair. |
| BETO head-tail | Cross-encoder variant preserving both beginning and end of long articles. |
| Rank10 cross-encoder | Regression model trained with graded relevance from the full ranking. |
| Textual ensemble | Weighted soft-voting combination of cross-encoder scores. |
| Tail reranking | Keeps the base top-1 and reranks the remaining candidates. |
| LLM ranking | Instruction-based ranking experiments with generative language models. |
| VLM fusion | Linear fusion between text scores and CLIP/SigLIP image-text scores. |

The final local textual ensemble uses:

```text
beto_headtail: 0.40
beto:          0.45
mdeberta:      0.15
```

## Reproducibility

The project uses a fixed random seed:

```python
SEED = 42
```

The custom split used for local experimentation is configured as:

```python
ACTIVE_DATASET = "tfg_split"
```

To regenerate the split, use:

```bash
python split_dataset.py
```

To inspect dataset statistics:

```bash
python dataset_metrics.py
```

To run error analysis over saved predictions:

```bash
python analyze_ranking_errors.py
```

## Testing the repository

A basic validation workflow is:

```bash
python -m compileall .
python evaluate_experiments.py --stage individual --models tfidf,bm25 --limit 5
```

If trained checkpoints are available:

```bash
python evaluate_experiments.py --stage ensembles \
  --ensemble-specs "beto_headtail:0.40,beto:0.45,mdeberta:0.15" \
  --limit 5
```

## Notes

- The code uses `beto` as the internal key for the BETO model, while the Hugging Face identifier remains `dccuchile/bert-base-spanish-wwm-cased`.
- `bertin` and `mdeberta` are separate model keys and should not be renamed.
- Dataset files, checkpoints, generated submissions and model weights are intentionally ignored by Git.
- LLM experiments may require a Hugging Face token and accepted access to gated models.
