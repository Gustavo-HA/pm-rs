"""
Barrido del peso alpha en el hibrido CFAttention + CBQuality.

    r_hat(alpha) = alpha * r_hat_CFAttention + (1 - alpha) * r_hat_CBQuality

Para cada alpha de la grilla se evaluan metricas top-K (NDCG, Coverage,
Hit, Precision, Recall, MRR, ILD) y se trackean en MLflow dentro de un
unico run usando `step = int(round(alpha * STEP_SCALE))`, de modo que los
charts de MLflow tienen alpha como eje X. Los nombres de las metricas se
sufijan con _at_K para separar curvas por K (e.g., ndcg_at_10).
La hibridacion se fitea una sola vez (precomputa el score_stack); el barrido
solo reasigna pesos via `set_weights`, evitando refits.

Persistencia: el HybridFusion ajustado y los dos modelos base se serializan
con joblib para inferencia posterior (ver scripts/infer_hybrid.py).

Uso:
    uv run python scripts/run_hybrid_alpha.py
    uv run python scripts/run_hybrid_alpha.py --k 5 10 20 --alpha-grid 0 0.25 0.5 0.75 1
    uv run python scripts/run_hybrid_alpha.py --experiment pm-rs-alpha-v2
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from mtrs.eval.metrics import evaluate_model
from mtrs.models import CFAttention, CBQuality, HybridFusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/rest-mex")
MATRICES_DIR = DATA_DIR / "matrices"
SPLITS_DIR = DATA_DIR / "splits"
DEFAULT_SAVE_DIR = Path("models/hybrid_ab")

# Factor para mapear alpha in [0, 1] a un step entero en MLflow.
# Permite grillas de hasta 1001 puntos (linspace(0, 1, 1001)) sin colisiones.
STEP_SCALE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Barrido de alpha en hibrido CFAttention + CBQuality."
    )
    parser.add_argument(
        "--matrices-dir",
        type=Path,
        default=MATRICES_DIR,
        help=f"Directorio con matrices A, X, Y (default: {MATRICES_DIR}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[10],
        help="Valores de K para top-K (default: 10).",
    )
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=None,
        help="Valores de alpha a evaluar (default: linspace(0, 1, 51)).",
    )
    parser.add_argument(
        "--beta-grid",
        type=float,
        nargs="+",
        default=None,
        help="Valores de β para la frontera α*(β)=argmax_α [β·NDCG@K+(1-β)·Cov@K] "
        "(default: linspace(0, 1, 21)).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="pm-rs-alpha",
        help="Nombre del experimento en MLflow (default: pm-rs-alpha).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefijo para el nombre del parent run.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=20,
        help="Vecinos para CFAttention.",
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.0,
        help="Similitud minima para CFAttention.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help=f"Directorio para guardar joblibs del hibrido (default: {DEFAULT_SAVE_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV de salida con resultados (opcional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha_grid = (
        np.asarray(args.alpha_grid, dtype=float)
        if args.alpha_grid is not None
        else np.linspace(0.0, 1.0, 51)
    )

    matrices_dir = args.matrices_dir
    logger.info(f"Usando matrices en {matrices_dir}")

    # ── Cargar matrices ──────────────────────────────────────────────────── #
    logger.info("Cargando matrices…")
    A = pd.read_parquet(matrices_dir / "A.parquet")
    X = pd.read_parquet(matrices_dir / "X.parquet")
    Y = pd.read_parquet(matrices_dir / "Y.parquet")
    logger.info(f"  A: {A.shape}  X: {X.shape}  Y: {Y.shape}")

    # ── Ground truth ────────────────────────────────────────────────────── #
    logger.info("Construyendo ground truth desde splits/test.parquet…")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")
    test_df = test_df[test_df["Author"].isin(A.index)]
    ground_truth: dict[str, set[str]] = (
        test_df.groupby("Author")["Pueblo"].apply(set).to_dict()
    )
    n_test_users = len(ground_truth)
    logger.info(f"  {n_test_users:,} usuarios de test con pueblos en A")

    # ── Fit modelos base ─────────────────────────────────────────────────── #
    logger.info("Entrenando CFAttention…")
    cba = CFAttention(k_neighbors=args.k_neighbors, min_sim=args.min_sim)
    t0 = time.perf_counter()
    cba.fit(X, A)
    cba_fit_time = time.perf_counter() - t0
    logger.info(f"  CFAttention fit: {cba_fit_time:.1f}s")

    logger.info("Entrenando CBQuality…")
    cbq = CBQuality()
    t0 = time.perf_counter()
    cbq.fit(X, Y, A)
    cbq_fit_time = time.perf_counter() - t0
    logger.info(f"  CBQuality fit: {cbq_fit_time:.1f}s")

    # ── Fit hibrido (precomputa score_stack para los dos modelos) ────────── #
    logger.info("Entrenando HybridFusion (precomputando score_stack)…")
    hybrid = HybridFusion({"CFAttention": cba, "CBQuality": cbq})
    t0 = time.perf_counter()
    hybrid.fit(A)
    hybrid_fit_time = time.perf_counter() - t0
    logger.info(f"  HybridFusion fit: {hybrid_fit_time:.1f}s")

    # ── Datasets MLflow ──────────────────────────────────────────────────── #
    train_dataset = mlflow.data.from_pandas(
        train_df,
        source=str(SPLITS_DIR / "train.parquet"),
        name="train",
    )
    test_dataset = mlflow.data.from_pandas(
        test_df,
        source=str(SPLITS_DIR / "test.parquet"),
        name="test",
    )

    dataset_params = {
        "A_shape": str(A.shape),
        "X_shape": str(X.shape),
        "Y_shape": str(Y.shape),
        "n_test_users": n_test_users,
    }

    # ── Persistencia ─────────────────────────────────────────────────────── #
    args.save_dir.mkdir(parents=True, exist_ok=True)
    hybrid_path = args.save_dir / "hybrid.joblib"
    cba_path = args.save_dir / "cfattention.joblib"
    cbq_path = args.save_dir / "cbquality.joblib"
    joblib.dump(hybrid, hybrid_path)
    joblib.dump(cba, cba_path)
    joblib.dump(cbq, cbq_path)
    logger.info(f"Modelos serializados en {args.save_dir}")

    # ── MLflow ───────────────────────────────────────────────────────────── #
    mlflow.set_tracking_uri("http://0.0.0.0:1825")
    mlflow.set_experiment(args.experiment)

    parent_run_name = f"{args.prefix}_alpha_sweep" if args.prefix else "alpha_sweep"

    all_results: list[dict] = []

    with mlflow.start_run(run_name=parent_run_name):
        # Datasets + params globales
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(test_dataset, context="testing")
        mlflow.log_params(dataset_params)
        mlflow.log_params(
            {
                "base_models": "CFAttention,CBQuality",
                "alpha_grid": ",".join(f"{a:.4f}" for a in alpha_grid),
                "n_alphas": len(alpha_grid),
                "k_values": ",".join(str(k) for k in args.k),
                **{f"cba_{k}": v for k, v in cba.get_params().items()},
                **{f"cbq_{k}": v for k, v in cbq.get_params().items()},
            }
        )
        mlflow.log_metric("cba_fit_time_sec", cba_fit_time)
        mlflow.log_metric("cbq_fit_time_sec", cbq_fit_time)
        mlflow.log_metric("hybrid_fit_time_sec", hybrid_fit_time)

        # Artefactos serializados
        mlflow.log_artifact(str(hybrid_path))
        mlflow.log_artifact(str(cba_path))
        mlflow.log_artifact(str(cbq_path))

        mlflow.log_param("step_scale", STEP_SCALE)

        # ── Sweep ──────────────────────────────────────────────────────── #
        # step = round(alpha * STEP_SCALE) → eje X = alpha en los charts.
        # Metrica `alpha` re-logueada en cada step para deshacer el escalado en hover.
        for alpha in alpha_grid:
            alpha = float(alpha)
            step = int(round(alpha * STEP_SCALE))
            hybrid.set_weights(np.array([alpha, 1.0 - alpha]))

            metrics_to_log: dict[str, float] = {"alpha": alpha}

            for k in args.k:
                t0 = time.perf_counter()
                metrics = evaluate_model(hybrid, ground_truth, k=k, Y=Y)
                eval_time = time.perf_counter() - t0

                logger.info(
                    f"  alpha={alpha:.2f}  "
                    f"NDCG@{k}={metrics['ndcg']:.4f}  "
                    f"Cov@{k}={metrics['coverage']:.4f}  "
                    f"P@{k}={metrics['precision']:.4f}  "
                    f"R@{k}={metrics['recall']:.4f}  "
                    f"MRR@{k}={metrics['mrr']:.4f}  "
                    f"ILD@{k}={metrics['ild']:.4f}  "
                    f"({eval_time:.1f}s)"
                )

                metrics_to_log.update(
                    {
                        f"ndcg_at_{k}": metrics["ndcg"],
                        f"coverage_at_{k}": metrics["coverage"],
                        f"hit_at_{k}": metrics["hit"],
                        f"precision_at_{k}": metrics["precision"],
                        f"recall_at_{k}": metrics["recall"],
                        f"mrr_at_{k}": metrics["mrr"],
                        f"ild_at_{k}": metrics["ild"],
                        f"eval_time_sec_at_{k}": eval_time,
                    }
                )

                all_results.append({"alpha": alpha, "K": k, **metrics})

            mlflow.log_metrics(metrics_to_log, step=step)

    # ── Resumen ──────────────────────────────────────────────────────────── #
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("BARRIDO DE ALPHA — RESULTADOS")
    print("=" * 80)
    for k in args.k:
        sub = results_df[results_df["K"] == k].set_index("alpha").drop(columns="K")
        sub.columns = [
            f"{c}@{k}" if c not in ("ild", "coverage") else c for c in sub.columns
        ]
        print(f"\n── K = {k} ──")
        print(sub.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── Frontera α*(β) ───────────────────────────────────────────────────── #
    # El óptimo del objetivo escalarizado f(α)=β·NDCG@K+(1-β)·Cov@K se obtiene
    # por barrido: α*(β) = argmax_α f(α). No se requiere metaheurística porque
    # hay un solo parámetro escalar α ∈ [0, 1].
    beta_grid = (
        np.asarray(args.beta_grid, dtype=float)
        if args.beta_grid is not None
        else np.linspace(0.0, 1.0, 21)
    )
    frontier: list[dict] = []
    for k in args.k:
        sub = results_df[results_df["K"] == k]
        alphas = sub["alpha"].to_numpy()
        ndcg = sub["ndcg"].to_numpy()
        cov = sub["coverage"].to_numpy()
        for beta in beta_grid:
            f = beta * ndcg + (1.0 - beta) * cov
            i = int(np.argmax(f))
            frontier.append(
                {
                    "K": k,
                    "beta": float(beta),
                    "alpha_star": float(alphas[i]),
                    "f": float(f[i]),
                    "ndcg": float(ndcg[i]),
                    "coverage": float(cov[i]),
                }
            )
    frontier_df = pd.DataFrame(frontier)

    print("\n" + "=" * 80)
    print("FRONTERA α*(β) — argmax_α [β·NDCG@K + (1-β)·Cov@K]")
    print("=" * 80)
    for k in args.k:
        print(f"\n── K = {k} ──")
        print(
            frontier_df[frontier_df["K"] == k]
            .drop(columns="K")
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output, index=False)
        frontier_path = args.output.with_name(args.output.stem + "_frontier.csv")
        frontier_df.to_csv(frontier_path, index=False)
        logger.info(f"\nResultados guardados en {args.output}")
        logger.info(f"Frontera α*(β) guardada en {frontier_path}")


if __name__ == "__main__":
    main()
