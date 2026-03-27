# Interfaz base para todos los modelos del sistema de recomendación.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self


class BaseRecommender(ABC):
    """ABC con interfaz unificada: fit / predict / recommend."""

    # ------------------------------------------------------------------ #
    # API pública                                                          #
    # ------------------------------------------------------------------ #

    def get_params(self) -> dict[str, object]:
        """Devuelve los hiperparámetros del modelo (estilo sklearn)."""
        import inspect

        init = inspect.signature(self.__class__.__init__)
        return {
            name: getattr(self, name)
            for name, param in init.parameters.items()
            if name != "self"
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        }

    @abstractmethod
    def fit(self, *args, **kwargs) -> Self:
        """Entrena el modelo con las matrices disponibles."""
        ...

    @abstractmethod
    def predict(self, user: str, pueblo: str) -> float:
        """Predice r̂_{u,p} ∈ [1, 5]."""
        ...

    def recommend(
        self,
        user: str,
        k: int = 10,
        exclude_visited: bool = True,
    ) -> list[tuple[str, float]]:
        """Devuelve los top-K pueblos con mayor r̂_{u,p}.

        Args:
            user: identificador del usuario (Author).
            k: número de recomendaciones.
            exclude_visited: si True, excluye pueblos ya calificados en A.
        """
        candidates = list(self._all_pueblos)
        if exclude_visited:
            visited = self._visited(user)
            candidates = [p for p in candidates if p not in visited]

        scores = [(p, self.predict(user, p)) for p in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    # ------------------------------------------------------------------ #
    # Hooks para subclases                                                 #
    # ------------------------------------------------------------------ #

    @property
    def _all_pueblos(self) -> list[str]:
        raise NotImplementedError

    def _visited(self, user: str) -> set[str]:
        raise NotImplementedError
