# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thesis project: **"Sistema de Recomendación Híbrido para Pueblos Mágicos: Análisis de Aspectos y Optimización Metaheurística"** — Hybrid Recommender System for Mexican Pueblos Mágicos using the REST-MEX dataset (79,264 reviews, 9,582 users, 48 Pueblos Mágicos). Sparsity: 90.32%; strong rating imbalance (mean: 4.41, 61.3% five-star).

The thesis document lives in `reports/Tesis/` (main file: `gha_tesis_26.tex`, chapters in `chapters/`, bibliography in `references.bib`).

## Environment & Commands

- **Package manager**: `uv` — use `uv sync` to install all deps (including dev group)
- **Run python code**: `uv run script.py`
- **Run notebook**: `uv run jupyter notebook`
- **Add dependency**: `uv add <package>` / `uv add --group dev <package>`
- **Data versioning**: DVC with Google Drive remote — `dvc pull` to fetch data
- **GPU**: CUDA-enabled inference required for transformer pipelines (`--device cuda`)

### Pipeline Scripts

Run in order from `scripts/`:

```bash
uv run scripts/split_dataset.py
uv run scripts/run_aspects.py --splitter punkt --batch-size 64 --device cuda
uv run scripts/build_matrices.py
uv run scripts/run_models.py --k 5 10 20
```

## Data Pipeline

```
filtered_dataset.parquet
  → split_dataset.py      → splits/{train,test,excluded}.parquet   (last-month-out per user)
  → run_aspects.py        → absa/{punkt|char}_aspect_sentiment.parquet
  → build_matrices.py     → matrices/{A,X,Y,R}.parquet
  → run_models.py         → evaluation metrics (P@K, R@K, NDCG@K, MRR, ILD, Coverage)
```

**Working dataset** (`data/rest-mex/processed/filtered_dataset.parquet`): users with ≥3 distinct pueblos, pueblos with ≥10 distinct places. Schema: `Author`, `Titulo`, `Review`, `Calificacion` (1–5 float), `FechaEstadia` (datetime), `Pueblo`, `Estado`, `Tipo` (Hotel/Restaurant/Attractive), `Lugar`.

All data and notebooks are DVC-tracked.

## mtrs/ Package Architecture

```
mtrs/
├── aspects/        # ABSA pipeline
│   ├── config.py       # ASPECTS_BY_TYPE, ALL_ASPECTS, model names, hypothesis template
│   ├── splitters.py    # split_review_punkt(), split_review_char()
│   └── extractor.py    # AspectSentimentExtractor — orchestrates both HF pipelines
├── aggregation/    # Matrix computation
│   └── matrices.py     # compute_rating_matrix(), compute_user_aspect_importance(),
│                       # compute_pueblo_aspect_quality(), compute_user_pueblo_aspect_sentiment()
└── models/         # Recommender algorithms
    ├── base.py         # BaseRecommender ABC: fit(), predict(), recommend()
    ├── cf_classic.py   # CFClassic — SVD with biases (Koren et al. 2009)
    ├── cf_multicriteria.py  # CFMultiCriteria — aspect-aware user-user CF (Musto et al. 2017)
    ├── cb_quality.py   # CBQuality — content-based via aspect quality (Zhang et al. 2014)
    └── cb_attention.py # CBAttention — user-user CF via cosine similarity on X matrix
```

### Key Mathematical Formulations

- **CFClassic**: $\hat{r}_{u,p} = \mu + b_u + b_p + \mathbf{p}_u \cdot \mathbf{q}_p$ (SGD + L2 reg)
- **CBQuality**: $\hat{r}_{u,p} = \sum_j X_{u,j} Y_{p,j} / \sum_j X_{u,j}$
- **CBAttention**: $\hat{r}_{u,p} = \mu_u + \sum_v \text{sim}(u,v)(A_{v,p} - \mu_v) / \sum_v |\text{sim}(u,v)|$ where $\text{sim} = \cos(\mathbf{X}_u, \mathbf{X}_v)$

### ABSA Models

- **Aspect**: `Recognai/bert-base-spanish-wwm-cased-xnli` (zero-shot NLI classification)
- **Sentiment**: `vg055/roberta-base-bne-finetuned-TripAdvisorDomainAdaptation` (5-class: Muy Negativo → Muy Positivo)
- **Aspects by type** — Restaurant: [servicio, precio, ambiente, comida]; Hotel: [servicio, precio, ambiente, ubicación, habitación]; Attractive: [servicio, ambiente, precio, naturaleza, cultura]

### Matrices

| Matrix | Shape | Description |
|--------|-------|-------------|
| $A$ | users × pueblos | Mean rating per user-pueblo pair |
| $X$ | users × aspects | User aspect importance (sigmoid-shifted frequency) |
| $Y$ | pueblos × aspects | Pueblo aspect quality (volume-weighted sentiment) |
| $R$ | (users × pueblos) × aspects | User-pueblo-aspect sentiment (MultiIndex) |

## Thesis (LaTeX)

- Main file: `reports/Tesis/gha_tesis_26.tex`
- Chapters: `chap_01.tex` (intro), `chap_01b.tex`, `chap_02.tex`, `chap_03.tex`, `chap_04.tex`, `conclusions.tex`, `appendix.tex`
- Bibliography: `references.bib`
- Custom styles: `CIMATpreamble.sty`, `mypreamble.sty`
- VS Code is configured to build with `latexmk-pygmentize` recipe

## Wiki Maintenance

The project wiki lives in `wiki/`. It is the shared memory and context platform for this project.

### Directory Structure

```
wiki/
├── index.md       ← Master catalog. Read first for any query. Update after every operation.
├── log.md         ← Append-only event log. Never edit past entries.
├── overview.md    ← Evolving synthesis. Keep current.
├── MEMORY.md      ← Session-persistent context: open questions, hot threads, active tasks.
├── DECISIONS.md   ← Architecture and design decision log.
├── sources/       ← One summary page per ingested source.
├── entities/      ← People, datasets, programs, systems.
├── concepts/      ← Ideas, algorithms, frameworks, themes.
└── analyses/      ← Query answers, comparisons, special outputs.
```

### Session Start Checklist

At the start of every session:
1. Read `wiki/log.md` (last 5 entries) to recall recent activity.
2. Read `wiki/index.md` for a mental map of current content.
3. Read `wiki/MEMORY.md` for open threads and pending decisions.
4. Confirm with the user what they'd like to do today.

### Operations

**Ingest** (new source or major code change):
1. Read source carefully.
2. Create/update `wiki/sources/<slug>.md`.
3. Create/update entity pages in `wiki/entities/`.
4. Create/update concept pages in `wiki/concepts/`.
5. Update `wiki/overview.md` if the synthesis meaningfully shifts.
6. Update `wiki/index.md` (add/edit row, keep alphabetically sorted).
7. Append entry to `wiki/log.md`.

**Query** (user asks a question):
1. Read `wiki/index.md` → identify relevant pages → read them.
2. Synthesize answer with citations to wiki pages / sources.
3. Offer to save non-trivial answers as `wiki/analyses/<slug>.md`.
4. If saved, update `wiki/index.md` and append to `wiki/log.md`.

**Lint** (health-check on request):
- Flag contradictions, stale claims, orphan pages, missing cross-references, data gaps.
- Produce a brief report and offer to fix issues one by one or in batch.

### Page Frontmatter

Every wiki page must include:
```yaml
---
title: "Page Title"
category: source | entity | concept | analysis | overview
tags: [tag1, tag2]
sources: [source-slug]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

### Log Format

```markdown
## [YYYY-MM-DD] <operation> | <title>

<One paragraph description.>

Pages touched: [[Page A]], [[Page B]]
```

### Cross-Reference Policy

- Every entity/concept mentioned in a source summary must link to its wiki page.
- Every entity/concept page must list which source pages reference it.
- Use relative links: `[Page Name](../concepts/page-name.md)`.

---

## Interaction Rules

- Act as a senior Data Science / RecSys specialist. Use technical terminology with mathematical and statistical rigor.
- Be concise — only elaborate when explicitly asked.
- All mathematical formulas, loss functions, and algorithmic formulations must use LaTeX (`$...$` inline, `$$...$$` blocks).
- When defining technical concepts, include a practical example applied to this dataset.
- If given a problem in LaTeX, respond in LaTeX.
