# Filtrado Colaborativo clásico (SVD) con SGD puro en NumPy.
#
# Implementa el modelo de factorización matricial de Koren et al. (2009):
#
#   r̂_{u,p} = μ + b_u + b_p + p_u · q_p
#
# con biases de usuario (b_u) e ítem (b_p) y vectores latentes p_u, q_p ∈ ℝ^k.
# Entrenamiento por SGD con regularización L2:
#
#   e_{u,p} = r_{u,p} - r̂_{u,p}
#   b_u ← b_u + γ(e_{u,p} - λ b_u)
#   b_p ← b_p + γ(e_{u,p} - λ b_p)
#   p_u ← p_u + γ(e_{u,p} q_p - λ p_u)
#   q_p ← q_p + γ(e_{u,p} p_u - λ q_p)
#
# Referencia: Koren et al. (2009) — Matrix Factorization Techniques for
#             Recommender Systems. IEEE Computer, 42(8):30-37.

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Self

from .base import BaseRecommender


class CFClassic(BaseRecommender):
    """CF basado en factorización matricial SVD con biases (Koren et al., 2009).

    Factoriza A_{u,p} ≈ μ + b_u + b_p + p_u · q_p
    donde p_u ∈ ℝ^k y q_p ∈ ℝ^k son vectores latentes de usuario y pueblo.

    Args:
        n_factors: dimensión del espacio latente k.
        n_epochs: épocas de SGD sobre los ratings observados.
        lr: tasa de aprendizaje γ.
        reg: coeficiente de regularización L2 λ.
        random_state: semilla para reproducibilidad.
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        random_state: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state

    # ------------------------------------------------------------------ #

    def fit(self, A: pd.DataFrame) -> Self:  # type: ignore[override]
        """Entrena sobre la matriz de ratings A (usuarios × pueblos).

        Los NaN en A se interpretan como ratings no observados y se omiten.

        Args:
            A: DataFrame (m × n) con Calificacion promedio por (Author, Pueblo).
        """
        self._pueblos: list[str] = list(A.columns)
        self._users: list[str] = list(A.index)
        self._user_idx = {u: i for i, u in enumerate(self._users)}
        self._pueblo_idx = {p: i for i, p in enumerate(self._pueblos)}

        self._visited_map: dict[str, set[str]] = {
            user: set(row.dropna().index)
            for user, row in A.iterrows()
        }

        # Construir lista de tripletas (ui, pi, r) observadas
        long = A.stack().reset_index()
        long.columns = pd.Index(["user", "item", "rating"])
        users_obs = long["user"].map(self._user_idx).values
        items_obs = long["item"].map(self._pueblo_idx).values
        ratings_obs = long["rating"].values.astype(np.float64)

        m, n = len(self._users), len(self._pueblos)
        k = self.n_factors
        rng = np.random.default_rng(self.random_state)

        self._mu = float(ratings_obs.mean())
        self._bu = np.zeros(m)
        self._bp = np.zeros(n)
        self._P = rng.normal(0, 0.1, (m, k))
        self._Q = rng.normal(0, 0.1, (n, k))

        lr, reg = self.lr, self.reg

        for _ in range(self.n_epochs):
            idx = rng.permutation(len(ratings_obs))
            for i in idx:
                u, p, r = int(users_obs[i]), int(items_obs[i]), ratings_obs[i]
                pred = (
                    self._mu
                    + self._bu[u]
                    + self._bp[p]
                    + self._P[u] @ self._Q[p]
                )
                e = r - pred
                self._bu[u] += lr * (e - reg * self._bu[u])
                self._bp[p] += lr * (e - reg * self._bp[p])
                pu = self._P[u].copy()
                self._P[u] += lr * (e * self._Q[p] - reg * pu)
                self._Q[p] += lr * (e * pu - reg * self._Q[p])

        return self

    # ------------------------------------------------------------------ #

    def predict(self, user: str, pueblo: str) -> float:
        """r̂_{u,p} = μ + b_u + b_p + p_u·q_p, clamped en [1, 5]."""
        ui = self._user_idx.get(user)
        pi = self._pueblo_idx.get(pueblo)
        if ui is None or pi is None:
            return self._mu

        score = (
            self._mu
            + self._bu[ui]
            + self._bp[pi]
            + self._P[ui] @ self._Q[pi]
        )
        return float(np.clip(score, 1.0, 5.0))

    # ------------------------------------------------------------------ #

    @property
    def _all_pueblos(self) -> list[str]:
        return self._pueblos

    def _visited(self, user: str) -> set[str]:
        return self._visited_map.get(user, set())
