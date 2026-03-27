# CF Multi-criterio basado en distancia Euclidiana multi-dimensional sobre R_{u,p,j}.
#
# Implementa el modelo User-User CF de Musto et al. (2017):
#
#   dist(u_i, u_k) = (1 / |I(u_i,u_k)|) · Σ_{p ∈ I} d(R(u_i,p), R(u_k,p))
#
#   d(R(u_i,p), R(u_k,p)) = √ Σ_j |R_j(u_i,p) - R_j(u_k,p)|²
#
# donde I(u_i,u_k) son los pueblos co-visitados y j corre sobre los aspectos
# con sentimiento disponible para AMBOS usuarios en ese pueblo.
#
# Predicción (weighted nearest-neighbor):
#
#   r̂(u,p) = Σ_{v ∈ N_k(u)} sim(u,v)·A_{v,p} / Σ_{v} |sim(u,v)|
#
# Referencias:
#   Musto et al. (2017) RecSys '17, doi:10.1145/3109859.3109905

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Self

from .base import BaseRecommender


class CFMultiCriteria(BaseRecommender):
    """CF multi-criterio (Musto et al., 2017).

    La similitud entre usuarios se calcula como el inverso de la distancia
    Euclidiana promedio en el espacio de aspectos sobre los pueblos
    co-visitados en R.  Las predicciones finales se obtienen por promedio
    ponderado de los k vecinos más cercanos sobre la matriz A.

    Args:
        k_neighbors: número de vecinos a considerar en la predicción.
        min_common: pueblos en común mínimos para considerar un par de usuarios.
    """

    def __init__(self, k_neighbors: int = 20, min_common: int = 1) -> None:
        self.k_neighbors = k_neighbors
        self.min_common = min_common

    # ------------------------------------------------------------------ #

    def fit(  # type: ignore[override]
        self,
        R: pd.DataFrame,
        A: pd.DataFrame,
    ) -> Self:
        """Construye el tensor de sentimientos y almacena A para predicciones.

        Args:
            R: DataFrame (m·n × p) MultiIndex (Author, Pueblo), columnas = aspectos.
               Valores en [-2, 2] (sent_score).
            A: DataFrame (m × n) — ratings promedio (Author × Pueblo).
        """
        self._pueblos: list[str] = list(A.columns)
        self._users: list[str] = list(A.index)
        self._n_users = len(self._users)
        self._n_pueblos = len(self._pueblos)
        self._aspects: list[str] = list(R.columns)
        self._n_aspects = len(self._aspects)

        self._user_idx = {u: i for i, u in enumerate(self._users)}
        self._pueblo_idx = {p: i for i, p in enumerate(self._pueblos)}

        # Ratings para predicción ponderada
        self._A = A.reindex(index=self._users, columns=self._pueblos)
        self._A_vals = self._A.values.copy()  # (m, n) float64

        # Tensor de sentimientos R: (m, n, p) — NaN donde no disponible
        R_tensor = np.full(
            (self._n_users, self._n_pueblos, self._n_aspects),
            np.nan,
            dtype=np.float32,
        )
        for (user, pueblo), row in R.iterrows():
            ui = self._user_idx.get(user)
            pi = self._pueblo_idx.get(pueblo)
            if ui is None or pi is None:
                continue
            R_tensor[ui, pi, :] = row.values.astype(np.float32)

        self._R_tensor = R_tensor  # (m, n, p)

        # Mapa de pueblos visitados (observados en A)
        self._visited_map: dict[str, set[str]] = {
            user: set(row.dropna().index) for user, row in A.iterrows()
        }

        # Cache de similaridad (se llena bajo demanda)
        self._sim_cache: dict[str, np.ndarray] = {}
        return self

    # ------------------------------------------------------------------ #

    def _similarity_vector(self, user: str) -> np.ndarray:
        """Devuelve el vector de similitud sim(user, v) para todos los v.

        Calcula y cachea las distancias Euclidianas multi-criterio de `user`
        contra el resto de los usuarios.

        Returns:
            np.ndarray de shape (n_users,) con sim ∈ [0, 1].
        """
        if user in self._sim_cache:
            return self._sim_cache[user]

        ui = self._user_idx.get(user)
        if ui is None:
            return np.zeros(self._n_users, dtype=np.float32)

        R_u = self._R_tensor[ui]  # (n, p) — fila del usuario objetivo
        R_all = self._R_tensor  # (m, n, p) — todos los usuarios

        # Diferencias por aspecto: (m, n, p)
        diff = R_all - R_u[np.newaxis, :, :]  # broadcast sobre eje de usuarios

        # Máscara: aspecto válido = ambos no-NaN
        valid_u = ~np.isnan(R_u)  # (n, p)
        valid_all = ~np.isnan(R_all)  # (m, n, p)
        valid = valid_all & valid_u[np.newaxis, :, :]  # (m, n, p)

        # Distancia Euclidiana por pueblo (solo aspectos válidos en par)
        diff_sq = np.where(valid, diff**2, 0.0)  # (m, n, p)
        pueblo_dist = np.sqrt(diff_sq.sum(axis=2))  # (m, n)

        # Pueblos co-visitados: al menos un aspecto válido común
        co_visited = valid.any(axis=2)  # (m, n) bool

        n_common = co_visited.sum(axis=1).astype(np.float32)  # (m,)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean_dist = np.where(
                n_common >= self.min_common,
                (pueblo_dist * co_visited).sum(axis=1) / n_common,
                np.inf,
            )

        # Similitud = 1 / (1 + dist); para inf → 0
        similarity = 1.0 / (1.0 + mean_dist)
        similarity[ui] = 0.0  # excluir self-similarity

        self._sim_cache[user] = similarity.astype(np.float32)
        return self._sim_cache[user]

    # ------------------------------------------------------------------ #

    def predict(self, user: str, pueblo: str) -> float:
        """r̂_{u,p} como promedio ponderado de k vecinos sobre A."""
        pi = self._pueblo_idx.get(pueblo)
        if pi is None:
            return 3.0

        sim = self._similarity_vector(user)

        # Solo vecinos con A_{v,p} observado
        a_col = self._A_vals[:, pi]  # (m,)
        observed = ~np.isnan(a_col)  # (m,) bool

        active_sim = sim * observed  # zero out unobserved
        if active_sim.sum() == 0:
            return float(np.nanmean(a_col)) if np.any(observed) else 3.0

        # Top-k vecinos
        top_k = np.argpartition(-active_sim, min(self.k_neighbors, observed.sum() - 1))[
            : self.k_neighbors
        ]
        w = active_sim[top_k]
        r = a_col[top_k]
        valid = ~np.isnan(r) & (w > 0)

        if not valid.any():
            return 3.0

        return float(np.dot(w[valid], r[valid]) / w[valid].sum())

    # ------------------------------------------------------------------ #

    @property
    def _all_pueblos(self) -> list[str]:
        return self._pueblos

    def _visited(self, user: str) -> set[str]:
        return self._visited_map.get(user, set())
