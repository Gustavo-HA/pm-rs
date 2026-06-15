# Filtrado Colaborativo (rama CF) usuario-usuario sobre la atención X_{u,j}.
#
# Combina la matriz de atención usuario-aspecto X (Zhang et al., 2014) con
# la matriz de ratings A para estimar r̂_{u,p}^{att} mediante vecinos cercanos
# en el espacio de preferencias de aspectos:
#
#   sim_att(u, v) = 1 / (1 + ‖X_u - X_v‖₂)
#
#   r̂_{u,p}^{att} = μ_u + Σ_{v ∈ N_k} sim(u,v)·(A_{v,p} - μ_v)
#                         / Σ_{v ∈ N_k} |sim(u,v)|
#
# La corrección por media μ_u reduce el sesgo sistemático de calificación
# por usuario (algunos usuarios califican siempre alto/bajo).
#
# Referencia:
#   Zhang et al. (2014) SIGIR '14, doi:10.1145/2600428.2609579

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from typing import Self

from .base import BaseRecommender


class CFAttention(BaseRecommender):
    """User-User CF con similitud coseno sobre la matriz de atención X.

    Precomputa la matriz de similitud completa (m × m) en fit() para
    predicciones en O(k).

    Args:
        k_neighbors: vecinos más similares a considerar.
        min_sim: umbral mínimo de similitud para incluir un vecino.
    """

    def __init__(self, k_neighbors: int = 20, min_sim: float = 0.0) -> None:
        self.k_neighbors = k_neighbors
        self.min_sim = min_sim

    # ------------------------------------------------------------------ #

    def fit(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        A: pd.DataFrame,
    ) -> Self:
        """Precomputa la matriz de similitud euclidiana (m × m) sobre X.

        Args:
            X: DataFrame (m × p), valores en [1, 5].  Atención usuario-aspecto.
            A: DataFrame (m × n), ratings promedio (Author × Pueblo).
        """
        self._users: list[str] = list(A.index)
        self._pueblos: list[str] = list(A.columns)
        self._user_idx = {u: i for i, u in enumerate(self._users)}

        # Alinear X con los usuarios de A
        X_aligned = X.reindex(index=self._users).fillna(1.0)

        X_vals = X_aligned.values

        # Convertir distancia euclidiana a similitud en (0, 1]: 1/(1+d)
        dist = cdist(X_vals, X_vals, metric="euclidean")
        self._sim_matrix = (1.0 / (1.0 + dist)).astype(np.float32)
        np.fill_diagonal(self._sim_matrix, 0.0)  # excluir self-sim

        # Ratings y medias de usuario
        self._A_vals = A.reindex(index=self._users, columns=self._pueblos).values.copy()
        self._mu = np.nanmean(self._A_vals, axis=1)  # (m,)

        # Mapa de visitas
        self._visited_map: dict[str, set[str]] = {
            user: set(row.dropna().index) for user, row in A.iterrows()
        }
        return self

    # ------------------------------------------------------------------ #

    def predict(self, user: str, pueblo: str) -> float:
        """r̂_{u,p}^{att} con corrección por media de usuario."""
        ui = self._user_idx.get(user)
        pi = self._pueblos.index(pueblo) if pueblo in self._pueblos else None
        if ui is None or pi is None:
            return 3.0

        sim_u = self._sim_matrix[ui]  # (m,)

        # Solo vecinos con A_{v,p} observado y similitud ≥ min_sim
        a_col = self._A_vals[:, pi]  # (m,)
        observed = ~np.isnan(a_col)
        active_sim = np.where(observed & (sim_u >= self.min_sim), sim_u, 0.0)

        if active_sim.sum() == 0:
            return float(self._mu[ui]) if not np.isnan(self._mu[ui]) else 3.0

        # Top-k por similitud
        n_candidates = int(observed.sum())
        k = min(self.k_neighbors, n_candidates)
        if k == 0:
            return float(self._mu[ui]) if not np.isnan(self._mu[ui]) else 3.0

        top_k = np.argpartition(-active_sim, k)[:k]
        w = active_sim[top_k]
        dev = a_col[top_k] - self._mu[top_k]  # A_{v,p} - μ_v
        valid = ~np.isnan(dev) & (w > 0)

        if not valid.any():
            return float(self._mu[ui]) if not np.isnan(self._mu[ui]) else 3.0

        mu_u = self._mu[ui]
        if np.isnan(mu_u):
            mu_u = 3.0

        return float(mu_u + np.dot(w[valid], dev[valid]) / w[valid].sum())

    # ------------------------------------------------------------------ #

    @property
    def sim_matrix(self) -> np.ndarray:
        """Matriz de similitud coseno (m × m) para análisis."""
        return self._sim_matrix

    @property
    def _all_pueblos(self) -> list[str]:
        return self._pueblos

    def _visited(self, user: str) -> set[str]:
        return self._visited_map.get(user, set())
