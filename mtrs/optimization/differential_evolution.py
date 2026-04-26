"""
Differential Evolution (DE) para búsqueda de pesos en Δ^{M-1}.

Usa la implementación de scipy.optimize.differential_evolution con
estrategia 'best1bin' y restricción lineal de suma-unitaria.

Variante DE/best/1/bin:
    mutante  : v_i = x_best + F·(x_r1 - x_r2)
    trial    : u_ij = v_ij  si rand < CR  ó  j = j_rand, else x_ij
    selección: x_i ← u_i  si f(u_i) ≤ f(x_i)

La restricción Σw = 1, w ≥ 0 se aplica vía LinearConstraint + bounds [0,1]^M,
delegando el manejo de factibilidad al motor de scipy.

Referencias:
    Storn & Price (1997), "Differential Evolution – A Simple and Efficient
    Heuristic for global Optimization over Continuous Spaces", J. Global Opt.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import LinearConstraint, differential_evolution

from .base import (
    OptimizationResult,
    WeightOptimizer,
    build_eval_context,
    evaluate_weights,
    project_simplex,
)

logger = logging.getLogger(__name__)


class DEOptimizer(WeightOptimizer):
    """
    Differential Evolution sobre Δ^{M-1} vía scipy.

    Args:
        maxiter:       iteraciones máximas (convergencia o límite).
        popsize:       multiplicador de población: N = popsize × M.
        mutation:      factor de escala F ∈ (0, 2) o rango (F_min, F_max).
        recombination: probabilidad de cruce CR ∈ [0, 1].
        strategy:      variante DE ('best1bin', 'rand1bin', etc.).
        tol:           tolerancia de convergencia relativa.
        random_state:  semilla para reproducibilidad.
        verbose:       si True, registra cada iteración vía callback.
    """

    def __init__(
        self,
        maxiter: int = 300,
        popsize: int = 15,
        mutation: float | tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        strategy: str = "best1bin",
        tol: float = 1e-5,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.maxiter = maxiter
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.strategy = strategy
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def optimize(self, hybrid, ground_truth, Y, k=10, metric="ndcg"):
        ctx = build_eval_context(hybrid, ground_truth, Y)
        M = len(hybrid._model_list)

        history: list[float] = []
        n_evals_counter = [0]

        def objective(weights: np.ndarray) -> float:
            val = evaluate_weights(weights, ctx, k, metric)
            n_evals_counter[0] += 1
            return -val

        def callback(intermediate_result) -> bool:
            best = -intermediate_result.fun
            history.append(best)
            if self.verbose and len(history) % 20 == 0:
                logger.info(
                    f"DE iter {len(history):3d}  "
                    f"best_{metric}@{k}={best:.4f}  "
                    f"evals={n_evals_counter[0]}"
                )
            return False  # no detener

        bounds = [(0.0, 1.0)] * M
        # Restricción lineal: Σ w_i = 1
        simplex_constraint = LinearConstraint(
            np.ones((1, M)), lb=1.0, ub=1.0
        )

        result = differential_evolution(
            func=objective,
            bounds=bounds,
            constraints=simplex_constraint,
            strategy=self.strategy,
            maxiter=self.maxiter,
            popsize=self.popsize,
            mutation=self.mutation,
            recombination=self.recombination,
            seed=self.random_state,
            tol=self.tol,
            callback=callback,
            polish=True,  # L-BFGS-B final sobre el mejor individuo
        )

        best_weights = project_simplex(result.x)
        best_score = evaluate_weights(best_weights, ctx, k, metric)

        if not history:
            history = [best_score]

        logger.info(
            f"DE finalizado: {metric}@{k}={best_score:.4f}  "
            f"convergió={result.success}  "
            f"evals={n_evals_counter[0]}  "
            f"w={best_weights.round(3)}"
        )

        return OptimizationResult(
            optimizer_name="DE",
            best_weights=best_weights,
            best_score=best_score,
            metric=metric,
            k=k,
            history=history,
            n_evals=n_evals_counter[0],
        )
