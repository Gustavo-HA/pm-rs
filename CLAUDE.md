# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thesis project: **"Sistema de Recomendaci&oacute;n H&iacute;brido para Pueblos M&aacute;gicos: An&aacute;lisis de Aspectos y Optimizaci&oacute;n Metaheur&iacute;stica"** (Hybrid Recommender System for Pueblos M&aacute;gicos: Aspect-Based Analysis and Metaheuristic Optimization).

Hybrid recommender system (Collaborative Filtering + Content-Based) for Mexican Pueblos M&aacute;gicos using the REST-MEX dataset (79,264 reviews, 9,582 users, 48 Pueblos M&aacute;gicos). Sparsity: 90.32%, strong geographic/rating imbalance (mean: 4.41, 61.3% are 5-star).

## Technical Context

- **NLP Pipeline (ABSA)**: Aspect-Based Sentiment Analysis via Zero-Shot classification. Models: `Recognai/zeroshot_selectra_medium` (aspect classification), `pysentimiento/robertuito-sentiment-analysis` (sentiment). Candidate aspects under experimentation: "servicio", "gastronom&iacute;a", "naturaleza", "cultura", "hospitalidad", "precio", "ambiente" (and variations).
- **Collaborative Filtering**: `Surprise` library for baseline recommendation algorithms.
- **Optimization**: Metaheuristic fusion tuning (PSO, Bayesian, Differential Evolution — approach-agnostic).
- **Evaluation**: NDCG@K, Recall@K, Precision@K, MRR, Diversity. Explainability (XAI) required.
- **Goal**: Personalized Top-K recommendations per user, mitigating cold-start and high sparsity.

## Environment & Commands

- **Python**: 3.12 (`.python-version`)
- **Package manager**: `uv` (lockfile: `uv.lock`, config: `pyproject.toml`)
- **Setup**: `uv sync` (installs all deps including dev group)
- **Add dependency**: `uv add <package>` / `uv add --group dev <package>`
- **Run notebook**: `uv run jupyter notebook` or use VS Code/ipykernel
- **Data versioning**: DVC with Google Drive remote (`dvc pull` to fetch data)
- **GPU**: CUDA-enabled inference expected for transformer pipelines

## Data

All data lives under `data/rest-mex/` and is DVC-tracked:
- `main_dataset.parquet` — Full cleaned dataset (306,854 reviews, 9 columns)
- `filtered_dataset.parquet` — Iteratively filtered for density (users with &ge;3 distinct pueblos, pueblos with &ge;10 distinct places). **This is the working dataset** (79,264 reviews).
- `unique_pueblos.csv` — Pueblo reference list

Schema (filtered_dataset): `Author`, `Titulo`, `Review`, `Calificacion` (1-5 float), `FechaEstadia` (datetime), `Pueblo`, `Estado`, `Tipo` (Hotel/Restaurant/Attractive), `Lugar`

Notebooks are also DVC-tracked (`notebooks.dvc`).

## Architecture & Progress

Project progress is tracked through notebooks in `notebooks/`:

1. **`dataset.ipynb`** — Data ingestion from raw CSVs, cleaning (drop columns, parse dates), and iterative density filtering (`filter_dense` function). Produces both parquet files.
2. **`stats.ipynb`** — Exploratory analysis: distributions, chi-square test, Correspondence Analysis (Pueblo vs Calificaci&oacute;n using `prince`).
3. **`word-cloud.ipynb`** — Per-pueblo word clouds using NLTK stopwords for keyword discovery.
4. **`zero-shot.ipynb`** — ABSA pipeline: `analyze_review()` chunks reviews (default 100 chars), classifies aspect (zero-shot) and sentiment per chunk. `analyze_reviews_batch()` provides batch inference with configurable `batch_size`.

The `mtrs/` package (Magic Towns Recommender System) is the target for productionized code. Currently has an empty `aspects/` subpackage.

## Interaction Rules

- Act as a senior Data Science / RecSys specialist. Use technical terminology with mathematical and statistical rigor.
- Be concise — only elaborate when explicitly asked.
- All mathematical formulas, loss functions, and algorithmic formulations must use LaTeX (`$...$` inline, `$$...$$` blocks).
- When defining technical concepts, include a practical example applied to this dataset.
- If given a problem in LaTeX, respond in LaTeX.
- Ignore `reports/Tesis` directory for now.
