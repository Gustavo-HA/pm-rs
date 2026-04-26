"""
Bayesian Optimization (BO) con surrogate GP para búsqueda en Δ^{M-1}.

Esquema secuencial modelo-basado (SMBO):
  1. Muestreo inicial: n_initial puntos vía Latin Hypercube en [0,1]^M,
     proyectados al simplex antes de evaluar.
  2. Ajuste del surrogate: GaussianProcessRegressor con kernel Matérn(ν=2.5).
  3. Maximización de la adquisición EI (Expected Improvement) con
     multi-start L-BFGS-B en [0,1]^M.
  4. Proyectar candidato → simplex, evaluar, actualizar GP.
  5. Repetir n_iter veces.

Expected Improvement:
    EI(x) = (μ(x) - f* - ξ) · Φ(Z) + σ(x) · φ(Z)
    Z = (μ(x) - f* - ξ) / σ(x)

donde f* = max y observado, ξ es el parámetro de exploración.

Referencias:
    Snoek et al. (2012), "Practical Bayesian Optimization of Machine
    Learning Algorithms", NeurIPS.
    Rasmussen & Williams (2006), "Gaussian Processes for Machine Learning".
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .base import (
    OptimizationResult,
    WeightOptimizer,
    build_eval_context,
    evaluate_weights,
    project_simplex,
)

logger = logging.getLogger(__name__)


def _expected_improvement(
    X_cand: np.ndarray,
    gp: GaussianProcessRegressor,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    EI(x) vectorizado para un conjunto de candidatos X_cand ∈ ℝ^{n×M}.

    Returns:
        ei: array (n,) con EI ≥ 0 para cada candidato.
    """
    mu, sigma = gp.predict(X_cand, return_std=True)
    mu = mu.ravel()
    sigma = np.maximum(sigma.ravel(), 1e-9)
    Z = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return np.maximum(ei, 0.0)


def _maximize_ei(
    gp: GaussianProcessRegressor,
    f_best: float,
    M: int,
    n_restarts: int = 20,
    xi: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Maximiza EI sobre [0,1]^M con multi-start L-BFGS-B.

    Args:
        n_restarts: número de inicios aleatorios del optimizador local.

    Returns:
        Candidato óptimo en [0,1]^M (sin proyectar al simplex aún).
    """
    if rng is None:
        rng = np.random.default_rng()

    best_x: np.ndarray | None = None
    best_ei = -np.inf

    bounds = [(0.0, 1.0)] * M

    # Warmup: evaluar candidatos aleatorios antes del gradiente
    X_rand = rng.uniform(0.0, 1.0, size=(200, M))
    ei_rand = _expected_improvement(X_rand, gp, f_best, xi)
    top_idx = np.argsort(ei_rand)[-n_restarts:]

    for idx in top_idx:
        x0 = X_rand[idx]
        result = minimize(
            fun=lambda x: -float(_expected_improvement(x.reshape(1, -1), gp, f_best, xi).item()),
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-9},
        )
        ei_val = -result.fun
        if ei_val > best_ei:
            best_ei = ei_val
            best_x = result.x

    return best_x if best_x is not None else rng.uniform(0.0, 1.0, size=M)


class BayesianOptimizer(WeightOptimizer):
    """
    Bayesian Optimization sobre Δ^{M-1} con surrogate GP + adquisición EI.

    El GP opera en el espacio [0,1]^M; la proyección al simplex se aplica
    dentro de la evaluación. Esto es equivalente a modelar el paisaje de
    la métrica directamente en coordenadas proyectadas.

    Args:
        n_initial:      número de evaluaciones iniciales (LHS).
        n_iter:         iteraciones de optimización secuencial.
        xi:             parámetro de exploración de EI (ξ > 0).
        n_restarts_acq: reinicios para maximizar EI vía L-BFGS-B.
        alpha:          nivel de ruido del GP (regularización numérica).
        random_state:   semilla para reproducibilidad.
        verbose:        si True, imprime progreso en cada iteración.
    """

    def __init__(
        self,
        n_initial: int = 12,
        n_iter: int = 50,
        xi: float = 0.01,
        n_restarts_acq: int = 20,
        alpha: float = 1e-6,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.n_initial = n_initial
        self.n_iter = n_iter
        self.xi = xi
        self.n_restarts_acq = n_restarts_acq
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose

    def optimize(self, hybrid, ground_truth, Y, k=10, metric="ndcg"):
        rng = np.random.default_rng(self.random_state)
        ctx = build_eval_context(hybrid, ground_truth, Y)
        M = len(hybrid._model_list)

        # ── Kernel: Matérn(ν=2.5) + amplitud ─────────────────────────── #
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(M),
            length_scale_bounds=(1e-2, 10.0),
            nu=2.5,
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=int(rng.integers(0, 2**31)),
        )

        # ── Muestreo inicial: Latin Hypercube en [0,1]^M ─────────────── #
        from scipy.stats.qmc import LatinHypercube

        sampler = LatinHypercube(d=M, seed=self.random_state)
        X_init = sampler.random(n=self.n_initial)  # (n_initial, M)

        X_obs: list[np.ndarray] = []
        y_obs: list[float] = []

        logger.info(f"BO: evaluando {self.n_initial} puntos iniciales (LHS)…")
        for x in X_init:
            score = evaluate_weights(x, ctx, k, metric)
            X_obs.append(x.copy())
            y_obs.append(score)

        history: list[float] = []
        f_best = max(y_obs)
        best_x = X_obs[int(np.argmax(y_obs))].copy()
        history.append(f_best)
        n_evals = self.n_initial

        logger.info(f"BO: mejor inicial {metric}@{k}={f_best:.4f}")

        # ── Bucle BO ──────────────────────────────────────────────────── #
        X_arr = np.array(X_obs)
        y_arr = np.array(y_obs)

        for t in range(self.n_iter):
            gp.fit(X_arr, y_arr)

            x_next = _maximize_ei(
                gp=gp,
                f_best=f_best,
                M=M,
                n_restarts=self.n_restarts_acq,
                xi=self.xi,
                rng=rng,
            )

            score_next = evaluate_weights(x_next, ctx, k, metric)
            n_evals += 1

            X_arr = np.vstack([X_arr, x_next])
            y_arr = np.append(y_arr, score_next)

            if score_next > f_best:
                f_best = score_next
                best_x = x_next.copy()

            history.append(f_best)

            if self.verbose and (t + 1) % 5 == 0:
                w_proj = project_simplex(best_x)
                logger.info(
                    f"BO iter {t+1:3d}/{self.n_iter}  "
                    f"best_{metric}@{k}={f_best:.4f}  "
                    f"w={w_proj.round(3)}"
                )

        best_weights = project_simplex(best_x)
        best_score = evaluate_weights(best_weights, ctx, k, metric)

        logger.info(
            f"BO finalizado: {metric}@{k}={best_score:.4f}  "
            f"evals={n_evals}  w={best_weights.round(3)}"
        )

        return OptimizationResult(
            optimizer_name="BayesianOpt",
            best_weights=best_weights,
            best_score=best_score,
            metric=metric,
            k=k,
            history=history,
            n_evals=n_evals,
        )
