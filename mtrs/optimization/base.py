"""
Interfaz base y utilidades de evaluación vectorizada para optimizadores de pesos.

La función objetivo es:
    f(w) = metric(HybridFusion(w), ground_truth)

donde w ∈ Δ^{M-1} (simplex estándar). Todos los optimizadores minimizan -f(w).

La evaluación vectorizada evita el bucle Python de evaluate_model():
en lugar de llamar a model.recommend() usuario por usuario, calcula
S = einsum('m,mup->up', w, score_stack) y aplica top-k sobre la matriz
completa, reduciendo el tiempo de evaluación de O(U·P) llamadas Python
a O(1) einsum + slicing numpy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Simplex projection                                                           #
# --------------------------------------------------------------------------- #


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Proyección euclidiana sobre Δ^{n-1} (Duchi et al., 2008)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_arr = np.where(u * np.arange(1, n + 1) > (cssv - 1.0))[0]
    if len(rho_arr) == 0:
        return np.ones(n) / n
    theta = (cssv[rho_arr[-1]] - 1.0) / (rho_arr[-1] + 1.0)
    return np.maximum(v - theta, 0.0)


# --------------------------------------------------------------------------- #
# Contexto de evaluación                                                       #
# --------------------------------------------------------------------------- #


def build_eval_context(
    hybrid,
    ground_truth: dict[str, set[str]],
    Y: pd.DataFrame,
) -> dict:
    """
    Pre-computa estructuras de indexación para la evaluación vectorizada.

    Extrae del HybridFusion ya ajustado:
      - score_stack  : S ∈ ℝ^{M×U×P}  (precomputado en fit())
      - test_user_rows: índices i en la dim U de los usuarios de test
      - visited_masks : máscara booleana (P,) de pueblos ya visitados
      - relevant_sets : set de índices pueblo relevantes por usuario
      - Y_norm        : matriz Y normalizada para ILD

    Returns un dict opaco pasado a evaluate_weights().
    """
    user_idx: dict[str, int] = hybrid._user_idx
    pueblo_idx: dict[str, int] = hybrid._pueblo_idx
    pueblos: list[str] = hybrid._pueblos

    test_user_rows: list[int] = []
    visited_masks: list[np.ndarray] = []
    relevant_sets: list[set[int]] = []

    for user, rel_pueblos in ground_truth.items():
        if user not in user_idx:
            continue
        u_i = user_idx[user]
        test_user_rows.append(u_i)

        if hybrid._A is not None and user in hybrid._A.index:
            row = hybrid._A.loc[user]
            visited = set(row.index[row.notna()])
        else:
            visited = set()

        visited_masks.append(np.array([p in visited for p in pueblos], dtype=bool))
        relevant_sets.append({pueblo_idx[p] for p in rel_pueblos if p in pueblo_idx})

    Y_aligned = Y.reindex(pueblos).fillna(0.0).values.astype(np.float32)
    norms = np.linalg.norm(Y_aligned, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Y_norm = (Y_aligned / norms).astype(np.float32)

    return {
        "score_stack": hybrid._score_stack,
        "test_user_rows": test_user_rows,
        "visited_masks": visited_masks,
        "relevant_sets": relevant_sets,
        "Y_norm": Y_norm,
        "n_pueblos": len(pueblos),
    }


# --------------------------------------------------------------------------- #
# Evaluación vectorizada                                                       #
# --------------------------------------------------------------------------- #

SUPPORTED_METRICS = frozenset(
    {"ndcg", "mrr", "hit", "precision", "recall", "ild", "coverage"}
)


def evaluate_weights(
    weights: np.ndarray,
    ctx: dict,
    k: int,
    metric: str,
) -> float:
    """
    Evalúa pesos de fusión sobre los usuarios de test de forma vectorizada.

    Complejidad: O(einsum) + O(U·k·log k) — sin bucles Python sobre (u, p).

    Args:
        weights: vector en ℝ^M (se proyecta a Δ^{M-1} internamente).
        ctx:     contexto construido por build_eval_context().
        k:       longitud de la lista de recomendación.
        metric:  métrica objetivo (ver SUPPORTED_METRICS).

    Returns:
        Valor escalar de la métrica (mayor = mejor en todas las métricas).
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Métrica '{metric}' no soportada. Use: {SUPPORTED_METRICS}")

    w = project_simplex(weights).astype(np.float32)
    S = np.einsum("m,mup->up", w, ctx["score_stack"])  # (U, P)

    per_user: list[float] = []
    all_recs: set[int] = set()

    for u_i, mask, rel_idx in zip(
        ctx["test_user_rows"], ctx["visited_masks"], ctx["relevant_sets"]
    ):
        row = S[u_i].copy()
        row[mask] = -np.inf

        # top-k: argpartition O(P) + argsort de k elementos O(k log k)
        if k >= len(row):
            top_idx = np.argsort(row)[::-1]
        else:
            part = np.argpartition(row, -k)[-k:]
            top_idx = part[np.argsort(row[part])[::-1]]

        all_recs.update(top_idx.tolist())

        if metric == "ndcg":
            dcg = sum(
                1.0 / np.log2(rank + 2)
                for rank, idx in enumerate(top_idx)
                if idx in rel_idx
            )
            ideal = min(k, len(rel_idx))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
            per_user.append(dcg / idcg if idcg > 0 else 0.0)

        elif metric == "mrr":
            val = 0.0
            for rank, idx in enumerate(top_idx):
                if idx in rel_idx:
                    val = 1.0 / (rank + 1)
                    break
            per_user.append(val)

        elif metric == "hit":
            per_user.append(1.0 if any(i in rel_idx for i in top_idx) else 0.0)

        elif metric == "precision":
            per_user.append(sum(1 for i in top_idx if i in rel_idx) / k)

        elif metric == "recall":
            if rel_idx:
                per_user.append(sum(1 for i in top_idx if i in rel_idx) / len(rel_idx))
            else:
                per_user.append(0.0)

        elif metric == "ild":
            vecs = ctx["Y_norm"][top_idx]
            n = len(top_idx)
            if n < 2:
                per_user.append(0.0)
            else:
                sim = vecs @ vecs.T
                triu = sim[np.triu_indices(n, k=1)]
                per_user.append(float(1.0 - triu.mean()))

    if metric == "coverage":
        return len(all_recs) / ctx["n_pueblos"] if ctx["n_pueblos"] else 0.0

    return float(np.mean(per_user)) if per_user else 0.0


# --------------------------------------------------------------------------- #
# Resultado de optimización                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class OptimizationResult:
    """Contenedor de salida de cualquier optimizador."""

    optimizer_name: str
    best_weights: np.ndarray
    best_score: float
    metric: str
    k: int
    history: list[float] = field(default_factory=list)
    n_evals: int = 0

    def __repr__(self) -> str:
        w = ", ".join(f"{wi:.4f}" for wi in self.best_weights)
        return (
            f"OptimizationResult({self.optimizer_name}, "
            f"{self.metric}@{self.k}={self.best_score:.4f}, "
            f"w=[{w}], evals={self.n_evals})"
        )


# --------------------------------------------------------------------------- #
# Interfaz abstracta                                                           #
# --------------------------------------------------------------------------- #


class WeightOptimizer(ABC):
    """Clase base para todos los optimizadores de pesos de fusión."""

    @abstractmethod
    def optimize(
        self,
        hybrid,
        ground_truth: dict[str, set[str]],
        Y: pd.DataFrame,
        k: int = 10,
        metric: str = "ndcg",
    ) -> OptimizationResult:
        """
        Busca w* = argmax_{w ∈ Δ^{M-1}} metric(HybridFusion(w), test).

        Args:
            hybrid:       HybridFusion ya ajustado (con score_stack cargado).
            ground_truth: {user: {pueblo, ...}} del split de test.
            Y:            DataFrame pueblos × aspectos para ILD.
            k:            longitud de lista para top-K.
            metric:       métrica objetivo ('ndcg', 'mrr', 'hit', ...).

        Returns:
            OptimizationResult con best_weights, best_score e historial.
        """

    def _build_objective(self, ctx: dict, k: int, metric: str):
        """Devuelve la función objetivo escalar para minimizar: -metric(w)."""

        def objective(weights: np.ndarray) -> float:
            return -evaluate_weights(weights, ctx, k, metric)

        return objective
