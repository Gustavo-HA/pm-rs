---
title: "Design Decisions"
updated: 2026-04-25
---

# Design Decisions

> Architectural and implementation decisions, their rationale, and current status. Append new decisions; never remove old ones — mark as superseded instead.

---

## Decision 1: Type-Specific Aspect Labels in Zero-Shot Inference

**Date**: 2026-04-25 (reconstructed from code)  
**Status**: Finalized

### Decision

The zero-shot aspect classifier receives only the aspects valid for the venue type (`Tipo`) of each chunk, rather than all 8 aspects at once.

```python
for tipo, group in sent_df.groupby("Tipo"):
    labels = ASPECTS_BY_TYPE.get(tipo)
    res = self.aspect_classifier(batch, candidate_labels=labels, ...)
```

### Rationale

Zero-shot NLI classifiers compete all candidate labels against each other (softmax over scores). Mixing `habitación` (Hotel-only) with `comida` (Restaurant-only) in a single classification call would dilute the probability mass and introduce irrelevant competition between type-exclusive aspects. Grouping by `Tipo` preserves semantic clarity and produces cleaner aspect assignment.

### Trade-off

Slightly more complex inference loop (one pass per `Tipo` rather than one global pass). Acceptable given the benefit.

---

## Decision 2: Auto-Derived Temperature T from Data Median

**Date**: 2026-04-25 (reconstructed from code)  
**Status**: Finalized

### Decision

Both $X$ and $Y$ matrix computations auto-derive the sigmoid temperature $T$ as the median of non-zero values:

- $X$: $T = \text{median}(t_{u,j} | t_{u,j} > 0)$ — typical mention count
- $Y$: $T = \text{median}(|h_{p,j} \cdot \phi_{p,j}| \neq 0)$ — typical signal magnitude

### Rationale

A fixed $T$ would need to be tuned per dataset. By anchoring $T$ to the data distribution's median, the inflection point of the sigmoid (where $X = (N+1)/2 = 3$, the neutral value) always corresponds to the "typical" user or pueblo. This makes the parameterization dataset-agnostic and removes one hyperparameter from the search space.

### Trade-off

If the data distribution is bimodal or heavily skewed, the median may not be a meaningful anchor. For REST-MEX this is acceptable given the broadly uniform mention-count distribution.

---

## Decision 3: Last-Month-Out Temporal Split

**Date**: 2026-04-25 (reconstructed from code)  
**Status**: Finalized

### Decision

Test set = pueblos whose first visit by a user falls in that user's most recent month-year.

### Rationale

- Temporal splits prevent data leakage (no future data used for training)
- Per-user granularity aligns with the leave-one-out tradition in RecSys evaluation
- "Last month" is coarser than "last item" and more natural for tourism (multiple items per trip)
- Users with 100% pueblos in test are excluded (no training data — cannot evaluate)

### Trade-off

Users with only one month of history are excluded, slightly reducing test coverage. Accepted as correct behavior (these users cannot be meaningfully evaluated).

---

## Decision 4: Equal-Weight Baseline for HybridFusion

**Date**: 2026-04-25 (reconstructed from code)  
**Status**: Interim — pending metaheuristic optimization

### Decision

Default fusion weights are uniform: $w_m = 1/M = 0.25$ for all 4 base models.

### Rationale

A principled baseline before optimization. Equal weights test whether the hybrid combination alone (without tuning) already outperforms individual models. Also serves as the starting point for metaheuristic search.

### Next Step

Thesis Objective 5: optimize $\mathbf{w}$ via Differential Evolution, PSO, and/or Bayesian Optimization. Decision 4 will be superseded once optimal weights are found.

---

## Decision 5: Punkt Splitter as Default

**Date**: 2026-04-25 (reconstructed from code)  
**Status**: Tentative — no formal comparison run yet

### Decision

`--splitter punkt` is the default in pipeline documentation and scripts.

### Rationale

NLTK's Punkt tokenizer is trained for Spanish sentence boundaries and produces semantically coherent chunks. Char-based splitting is faster but may cut mid-sentence.

### Open Question

No ablation comparing punkt vs char-based ABSA output quality has been formally run. Both outputs exist in `data/rest-mex/absa/`. Matrix variants (`zs-bert-tempdefault` vs `zs-bert-tempmedian`) complicate this comparison further.

---

## Decision 6: Separate Matrix Subdirectories for Variants

**Date**: 2026-04-25 (reconstructed from directory structure)  
**Status**: Observed pattern — not explicitly documented

### Decision

Different pipeline runs (aspect model variants, temperature strategies) store their matrices in separate subdirectories under `data/rest-mex/matrices/`.

### Rationale

Allows parallel comparison of pipeline variants without overwriting the canonical matrices. The root `matrices/` directory holds the current canonical version.

### Risk

It is currently unclear which subdirectory is "canonical" for thesis results. See [MEMORY.md](MEMORY.md) — Open Thread 2.
