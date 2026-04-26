"""
Particle Swarm Optimization (PSO) para búsqueda de pesos en Δ^{M-1}.

Formulación estándar (Kennedy & Eberhart, 1995) con:
  - Decaimiento lineal del peso de inercia ω: ω_max → ω_min
  - Componentes cognitivo y social con coeficientes c₁, c₂
  - Proyección al simplex después de cada actualización de posición

Actualización de velocidad:
    v_i ← ω·v_i + c₁·r₁⊙(p_i - x_i) + c₂·r₂⊙(g - x_i)

Actualización de posición (con proyección):
    x_i ← Π_{Δ}(x_i + v_i)

donde Π_{Δ} es la proyección euclidiana sobre el simplex estándar.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import (
    OptimizationResult,
    WeightOptimizer,
    build_eval_context,
    evaluate_weights,
    project_simplex,
)

logger = logging.getLogger(__name__)


class PSOOptimizer(WeightOptimizer):
    """
    PSO sobre el simplex Δ^{M-1}.

    Args:
        n_particles: tamaño del enjambre.
        n_iter:      número máximo de iteraciones.
        omega_max:   peso de inercia inicial (decae hasta omega_min).
        omega_min:   peso de inercia final.
        c1:          coeficiente cognitivo (atracción al mejor personal).
        c2:          coeficiente social (atracción al mejor global).
        random_state: semilla para reproducibilidad.
        verbose:     si True, imprime progreso cada 10 iteraciones.
    """

    def __init__(
        self,
        n_particles: int = 30,
        n_iter: int = 150,
        omega_max: float = 0.9,
        omega_min: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.omega_max = omega_max
        self.omega_min = omega_min
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state
        self.verbose = verbose

    def optimize(self, hybrid, ground_truth, Y, k=10, metric="ndcg"):
        rng = np.random.default_rng(self.random_state)
        ctx = build_eval_context(hybrid, ground_truth, Y)
        M = len(hybrid._model_list)
        N = self.n_particles

        # ── Inicialización ─────────────────────────────────────────────── #
        # Posiciones: muestras de Dirichlet(α=1) = uniforme sobre el simplex
        positions = rng.dirichlet(np.ones(M), size=N)  # (N, M)
        velocities = rng.uniform(-0.1, 0.1, size=(N, M))

        scores = np.array(
            [evaluate_weights(positions[i], ctx, k, metric) for i in range(N)]
        )

        personal_best_pos = positions.copy()
        personal_best_scores = scores.copy()

        g_idx = int(np.argmax(scores))
        global_best_pos = positions[g_idx].copy()
        global_best_score = float(scores[g_idx])

        history: list[float] = [global_best_score]
        n_evals = N

        # ── Bucle principal ────────────────────────────────────────────── #
        for t in range(self.n_iter):
            omega = self.omega_max - (self.omega_max - self.omega_min) * t / self.n_iter

            r1 = rng.uniform(0.0, 1.0, size=(N, M))
            r2 = rng.uniform(0.0, 1.0, size=(N, M))

            velocities = (
                omega * velocities
                + self.c1 * r1 * (personal_best_pos - positions)
                + self.c2 * r2 * (global_best_pos - positions)
            )
            positions = positions + velocities
            positions = np.apply_along_axis(project_simplex, 1, positions)

            new_scores = np.array(
                [evaluate_weights(positions[i], ctx, k, metric) for i in range(N)]
            )
            n_evals += N

            improved = new_scores > personal_best_scores
            personal_best_pos[improved] = positions[improved]
            personal_best_scores[improved] = new_scores[improved]

            best_idx = int(np.argmax(new_scores))
            if new_scores[best_idx] > global_best_score:
                global_best_score = float(new_scores[best_idx])
                global_best_pos = positions[best_idx].copy()

            history.append(global_best_score)

            if self.verbose and (t + 1) % 10 == 0:
                logger.info(
                    f"PSO iter {t+1:3d}/{self.n_iter}  "
                    f"ω={omega:.3f}  "
                    f"best_{metric}@{k}={global_best_score:.4f}  "
                    f"w={global_best_pos.round(3)}"
                )

        return OptimizationResult(
            optimizer_name="PSO",
            best_weights=project_simplex(global_best_pos),
            best_score=global_best_score,
            metric=metric,
            k=k,
            history=history,
            n_evals=n_evals,
        )
