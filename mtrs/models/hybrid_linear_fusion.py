"""
Modelo de fusión lineal convexa de los cuatro recomendadores base.

    ŷ_{u,p}(w) = Σ_m  w_m · ŷ_{u,p}^(m)

Los pesos w viven en el simplex estándar Δ^{M-1}:
    Σ_m w_m = 1,  w_m ≥ 0  ∀m

Durante fit() se precomputan las matrices de puntuación S_m ∈ ℝ^{U×P}
de cada modelo base, de modo que el bucle de optimización sólo requiere
una multiplicación vectorial (no llamadas individuales a predict).
"""

from __future__ import annotations
import numpy as np
from .base import BaseRecommender


class HybridFusion(BaseRecommender):
    def __init__(self, models, weights=None):
        self.models = models
        self._model_names = list(models.keys())
        self._model_list = list(models.values())
        M = len(self._model_list)
        raw = np.ones(M) / M if weights is None else np.asarray(weights, dtype=float)
        self.weights = self._project_simplex(raw)

        self._users = []
        self._pueblos = []

        self._user_idx = {}
        self._pueblo_idx = {}

        self._A = None
        self._score_stack = None

    def fit(self, A):
        """Precomputa S_m ∈ ℝ^{U×P} para cada modelo base."""
        self._users = A.index.tolist()
        self._pueblos = A.columns.tolist()
        self._user_idx = {u: i for i, u in enumerate(self._users)}
        self._pueblo_idx = {p: i for i, p in enumerate(self._pueblos)}
        self._A = A
        M, U, P = len(self._model_list), len(self._users), len(self._pueblos)
        stack = np.empty((M, U, P), dtype=np.float32)
        for m_idx, model in enumerate(self._model_list):
            for u_idx, u in enumerate(self._users):
                for p_idx, p in enumerate(self._pueblos):
                    stack[m_idx, u_idx, p_idx] = model.predict(u, p)
        self._score_stack = stack
        return self

    def predict(self, user, pueblo):
        """O(1) lookup — dot product over precomputed stack."""
        u_idx = self._user_idx.get(user)
        p_idx = self._pueblo_idx.get(pueblo)
        if u_idx is None or p_idx is None:
            return 3.0
        return float(
            np.clip(self.weights @ self._score_stack[:, u_idx, p_idx], 1.0, 5.0)
        )

    def fused_matrix(self, weights):
        """S(w) = Σ_m w_m S_m  via einsum — used by optimizer."""
        return np.einsum("m,mup->up", weights.astype(np.float32), self._score_stack)

    def set_weights(self, weights):
        self.weights = self._project_simplex(np.asarray(weights, dtype=float))

    @property
    def _all_pueblos(self):
        return self._pueblos

    def _visited(self, user):
        if self._A is None or user not in self._A.index:
            return set()
        row = self._A.loc[user]
        return set(row.index[row.notna()])

    @staticmethod
    def _project_simplex(v):
        """Euclidean projection onto Δ^{n-1}. Duchi et al. (2008)."""
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_arr = np.where(u * np.arange(1, n + 1) > (cssv - 1.0))[0]
        if len(rho_arr) == 0:
            return np.ones(n) / n
        theta = (cssv[rho_arr[-1]] - 1.0) / (rho_arr[-1] + 1.0)
        return np.maximum(v - theta, 0.0)
