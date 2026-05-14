"""Top-K recommender metrics: P@K, R@K, NDCG@K, MRR, Hit@K, ILD, Coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd


def hit_at_k(recs: list[str], relevant: set[str]) -> float:
    return 1.0 if any(r in relevant for r in recs) else 0.0


def precision_at_k(recs: list[str], relevant: set[str]) -> float:
    if not recs:
        return 0.0
    hits = sum(1 for r in recs if r in relevant)
    return hits / len(recs)


def recall_at_k(recs: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in recs if r in relevant)
    return hits / len(relevant)


def ndcg_at_k(recs: list[str], relevant: set[str]) -> float:
    """Binary relevance NDCG@K."""
    dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(recs) if r in relevant)
    ideal_hits = min(len(recs), len(relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(recs: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first hit, 0 if no hit."""
    for i, r in enumerate(recs):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def intra_list_diversity(recs: list[str], Y: pd.DataFrame) -> float:
    """Mean pairwise cosine distance between recommended pueblos in Y-space."""
    available = [p for p in recs if p in Y.index]
    if len(available) < 2:
        return 0.0
    vecs = Y.loc[available].values
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs_norm = vecs / norms
    sim_matrix = vecs_norm @ vecs_norm.T
    n = len(available)
    triu = sim_matrix[np.triu_indices(n, k=1)]
    return float(1.0 - triu.mean())


def evaluate_model(
    model,
    test_ground_truth: dict[str, set[str]],
    k: int,
    Y: pd.DataFrame,
) -> dict[str, float]:
    """Itera sobre usuarios en test_ground_truth, genera top-K y agrega metricas."""
    metrics_per_user: list[dict[str, float]] = []
    all_recommended: set[str] = set()

    for user, relevant in test_ground_truth.items():
        recs = [pueblo for pueblo, _ in model.recommend(user, k=k)]
        all_recommended.update(recs)

        metrics_per_user.append(
            {
                "hit": hit_at_k(recs, relevant),
                "precision": precision_at_k(recs, relevant),
                "recall": recall_at_k(recs, relevant),
                "ndcg": ndcg_at_k(recs, relevant),
                "mrr": mrr(recs, relevant),
                "ild": intra_list_diversity(recs, Y),
            }
        )

    agg = {
        metric: float(np.mean([m[metric] for m in metrics_per_user]))
        for metric in metrics_per_user[0]
    }
    n_pueblos = len(model._all_pueblos)
    agg["coverage"] = len(all_recommended) / n_pueblos if n_pueblos else 0.0
    return agg
