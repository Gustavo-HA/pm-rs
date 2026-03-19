# Content-Based: calidad pueblo-aspecto ponderada por atención del usuario.
#
# Implementa el modelo de scoring de Zhang et al. (2014) Eq.3 en modo CB puro:
#
#   r̂_{u,p}^{qual} = Σ_j X_{u,j} · Y_{p,j} / Σ_j X_{u,j}
#
# donde X_{u,j} ∈ [1,5] es la importancia del usuario u por el aspecto j
# e Y_{p,j} ∈ [1,5] es la calidad del pueblo p en ese aspecto.
#
# El denominador normaliza sobre los pesos de atención del usuario, produciendo
# una puntuación en [1, 5] interpretable como calidad esperada percibida.
#
# Referencia:
#   Zhang et al. (2014) SIGIR '14, doi:10.1145/2600428.2609579

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseRecommender


class CBQuality(BaseRecommender):
    """Content-Based por calidad ponderada (Zhang et al., 2014).

    Precomputa en fit() la matriz de scores completa (m × n) para O(1)
    en predict().

    Args:
        A: requerido en fit() únicamente para filtrar pueblos visitados
           en recommend().  No afecta los scores calculados.
    """

    def fit(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        A: pd.DataFrame | None = None,
    ) -> "CBQuality":
        """Precomputa r̂_{u,p}^{qual} = (X_u / ‖X_u‖₁) · Y_p para todos los pares.

        Args:
            X: DataFrame (m × p), valores en [1, 5].  Atención usuario-aspecto.
            Y: DataFrame (n × p), valores en [1, 5].  Calidad pueblo-aspecto.
            A: DataFrame (m × n) opcional; solo para filtrar visitados.
        """
        # Align aspect columns
        aspects = X.columns.intersection(Y.columns)
        X = X[aspects]
        Y = Y[aspects]

        # Normalización L1 por usuario: w_{u,j} = X_{u,j} / Σ_j X_{u,j}
        row_sums = X.sum(axis=1).replace(0, 1)   # evitar / 0
        W = X.div(row_sums, axis=0)              # (m, p) — pesos normalizados

        # Producto matricial: (m, p) · (p, n) → (m, n)
        scores_arr = W.values @ Y.values.T        # (m, n)

        self._scores = pd.DataFrame(
            scores_arr,
            index=X.index,
            columns=Y.index,
        )

        self._pueblos: list[str] = list(Y.index)

        # Mapa de visitas desde A (opcional)
        if A is not None:
            self._visited_map: dict[str, set[str]] = {
                user: set(row.dropna().index)
                for user, row in A.iterrows()
            }
        else:
            self._visited_map = {}

        return self

    # ------------------------------------------------------------------ #

    def predict(self, user: str, pueblo: str) -> float:
        """r̂_{u,p}^{qual} ∈ [1, 5]."""
        try:
            return float(self._scores.loc[user, pueblo])
        except KeyError:
            return 3.0

    # ------------------------------------------------------------------ #

    @property
    def scores(self) -> pd.DataFrame:
        """Matriz de scores completa (m × n) para análisis."""
        return self._scores

    @property
    def _all_pueblos(self) -> list[str]:
        return self._pueblos

    def _visited(self, user: str) -> set[str]:
        return self._visited_map.get(user, set())
