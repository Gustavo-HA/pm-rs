---
title: "Session Memory"
updated: 2026-04-25
---

# Session Memory

> Persistent context for Claude Code sessions on this project. Update as active threads change.

---

## Project Status

- **Phase**: Implementation complete for all 4 base models + HybridFusion. Weight optimization pending.
- **Next milestone**: Metaheuristic optimization of fusion weights (Objective 5 of thesis)
- **Current models**: CFClassic, CFMultiCriteria, CBQuality, CBAttention, HybridFusion (equal weights baseline)
- **Evaluation**: Logging to MLflow at `http://0.0.0.0:1825`

---

## Open Threads

### Thread 1: Metaheuristic Optimization

The thesis requires comparing Differential Evolution, PSO, and/or Bayesian Optimization for fusion weight tuning. No optimization script exists yet. The `HybridFusion.fused_matrix(weights)` method is the hook for this.

**Questions to resolve**:
- Which metrics to optimize? (NDCG for accuracy, ILD for diversity, or a linear combination?)
- Should we use the train set NDCG, or a validation split within train?
- DE vs PSO vs Bayesian — fairness of comparison requires same budget

### Thread 2: Aspect Model Version

`config.py` shows the aspect model was updated from `Recognai/zeroshot_selectra_medium` to `Recognai/bert-base-spanish-wwm-cased-xnli`. Multiple matrix variants exist in `data/rest-mex/matrices/` (zs-bert, zs-bert-tempdefault, zs-bert-tempmedian). It is unclear which variant is the current canonical one used in thesis results.

### Thread 3: Thesis Chapter 04

`chap_04.tex` is the experiments chapter — its current content was not fully read. Pending: read and confirm what results have already been written vs. what still needs to be run.

---

## Active Decisions

| # | Decision | Status |
|---|----------|--------|
| 1 | Type-specific aspect labels in zero-shot inference | Finalized |
| 2 | Auto-derived temperature T from data median | Finalized |
| 3 | Last-month-out temporal split | Finalized |
| 4 | Equal-weight baseline for HybridFusion | Interim (pending optimization) |
| 5 | punkt splitter as default (vs char-based) | Needs comparison |

See full rationale in [DECISIONS.md](DECISIONS.md).

---

## Useful Commands

```bash
# Full pipeline (from clean state)
uv run scripts/split_dataset.py
uv run scripts/run_aspects.py --splitter punkt --batch-size 64 --device cuda
uv run scripts/build_matrices.py
uv run scripts/run_models.py --k 5 10 20

# Just evaluate models (matrices already built)
uv run scripts/run_models.py --k 5 10 20 --experiment my-run

# Launch MLflow UI
mlflow ui --host 0.0.0.0 --port 1825
```

---

## Known Quirks

- `CFMultiCriteria` caches similarity vectors in `_sim_cache` — the cache is per-instance, not persisted. For large evaluations, this prevents redundant computation within a session.
- `HybridFusion.fit(A)` iterates over all `(m × n)` pairs calling `.predict()` on each base model — can be slow for large catalogs. Pre-fitting base models first is required.
- DVC data is on Google Drive remote — `dvc pull` required on fresh checkout.
