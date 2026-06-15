"""
Ablación sobre la construcción de X (importancia usuario-aspecto)
e Y (calidad pueblo-aspecto).

Cuatro condiciones cruzadas:

    Variante │ X                       │ Y
    ─────────┼─────────────────────────┼──────────────────────────────────
       A     │ sigmoide + T (actual)   │ h * log(1+n) + T (actual)
       B     │ Zhang Eq.2 puro (T=1)   │ Zhang Eq.3 puro (sin log, sin T)
       C     │ Zhang Eq.2 puro (T=1)   │ Zhang Eq.3 + T (sin log)
       D     │ sigmoide + T (actual)   │ Zhang Eq.3 + T (sin log)

Modelos evaluados (los que dependen de X y/o Y):
  - CFAttention  (depende de X)
  - CBQuality    (depende de X y de Y)

Comparaciones:
  A vs D → aporte de log1p en Y
  B vs C → aporte de T en Y
  A vs B → diferencia conjunta (sanity check)
  B vs D → aporte de la T en X y log1p en Y combinados

Uso:
    uv run python scripts/run_ablation_xy.py
    uv run python scripts/run_ablation_xy.py --k 5 10 20
    uv run python scripts/run_ablation_xy.py --variants A B C D
    uv run python scripts/run_ablation_xy.py --output reports/ablation_xy.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from mtrs.aggregation.matrices import (
    compute_pueblo_aspect_quality,
    compute_pueblo_aspect_quality_zhang,
    compute_pueblo_aspect_quality_zhang_T,
    compute_rating_matrix,
    compute_user_aspect_importance,
    compute_user_aspect_importance_zhang,
)
from mtrs.eval.metrics import evaluate_model
from mtrs.models import CFAttention, CBQuality

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/rest-mex")
DEFAULT_ABSA = DATA_DIR / "absa" / "punkt_bert_absa.parquet"
SPLITS_DIR = DATA_DIR / "splits"


# --------------------------------------------------------------------------- #
# Variantes                                                                    #
# --------------------------------------------------------------------------- #


VARIANTS: dict[str, dict[str, str]] = {
    "A": {
        "X": "sigmoid+T (actual)",
        "Y": "h*log(1+n)+T (actual)",
        "x_fn": "compute_user_aspect_importance",
        "y_fn": "compute_pueblo_aspect_quality",
    },
    "B": {
        "X": "Zhang Eq.2 puro",
        "Y": "Zhang Eq.3 puro",
        "x_fn": "compute_user_aspect_importance_zhang",
        "y_fn": "compute_pueblo_aspect_quality_zhang",
    },
    "C": {
        "X": "Zhang Eq.2 puro",
        "Y": "Zhang Eq.3 + T",
        "x_fn": "compute_user_aspect_importance_zhang",
        "y_fn": "compute_pueblo_aspect_quality_zhang_T",
    },
    "D": {
        "X": "sigmoid+T (actual)",
        "Y": "Zhang Eq.3 + T",
        "x_fn": "compute_user_aspect_importance",
        "y_fn": "compute_pueblo_aspect_quality_zhang_T",
    },
}


def build_x(variant: str, df_absa: pd.DataFrame) -> pd.DataFrame:
    if VARIANTS[variant]["x_fn"] == "compute_user_aspect_importance":
        return compute_user_aspect_importance(df_absa)
    return compute_user_aspect_importance_zhang(df_absa)


def build_y(variant: str, df_absa: pd.DataFrame) -> pd.DataFrame:
    fn = VARIANTS[variant]["y_fn"]
    if fn == "compute_pueblo_aspect_quality":
        return compute_pueblo_aspect_quality(df_absa)
    if fn == "compute_pueblo_aspect_quality_zhang":
        return compute_pueblo_aspect_quality_zhang(df_absa)
    return compute_pueblo_aspect_quality_zhang_T(df_absa)


# --------------------------------------------------------------------------- #
# Diagnóstico de matrices                                                      #
# --------------------------------------------------------------------------- #


def matrix_diagnostics(M: pd.DataFrame, name: str) -> dict[str, float]:
    """Estadísticos clave para detectar saturación / colapso."""
    vals = M.values.astype(np.float64)
    return {
        f"{name}_mean": float(np.nanmean(vals)),
        f"{name}_std": float(np.nanstd(vals)),
        f"{name}_row_var_mean": float(np.nanmean(np.nanvar(vals, axis=1))),
        f"{name}_frac_saturated_high": float(np.nanmean(vals > 4.5)),
        f"{name}_frac_saturated_low": float(np.nanmean(vals < 1.5)),
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablación de las matrices X e Y (4 variantes cruzadas)."
    )
    parser.add_argument(
        "--absa",
        type=Path,
        default=DEFAULT_ABSA,
        help=f"Parquet con salida ABSA (default: {DEFAULT_ABSA}).",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        choices=list(VARIANTS.keys()),
        default=list(VARIANTS.keys()),
        help="Variantes a evaluar (default: A B C D).",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[10],
        help="Valores de K para top-K (default: 10).",
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
        help="Similitud mínima para CFAttention.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV de salida con resultados (opcional).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="ablacion_xy",
        help="Nombre del experimento en MLflow (default: ablacion_xy).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefijo para nombres de run en MLflow.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()

    # ── Cargar ABSA y splits ─────────────────────────────────────────────── #
    logger.info(f"Cargando ABSA desde {args.absa}…")
    df_absa = pd.read_parquet(args.absa)
    logger.info(f"  ABSA: {len(df_absa):,} filas")

    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")

    # ── Matrices invariantes (A no depende de la variante) ──────────────── #
    logger.info("Construyendo A (rating matrix) — invariante…")
    A = compute_rating_matrix(df_absa)
    logger.info(f"  A: {A.shape}")

    # Filtrar test users a los presentes en A
    test_df = test_df[test_df["Author"].isin(A.index)]
    ground_truth: dict[str, set[str]] = (
        test_df.groupby("Author")["Pueblo"].apply(set).to_dict()
    )
    n_test_users = len(ground_truth)
    logger.info(f"  {n_test_users:,} usuarios de test")

    # ── MLflow setup ─────────────────────────────────────────────────────── #
    mlflow.set_tracking_uri("http://0.0.0.0:1825")
    mlflow.set_experiment(args.experiment)

    train_dataset = mlflow.data.from_pandas(
        train_df, source=str(SPLITS_DIR / "train.parquet"), name="train"
    )
    test_dataset = mlflow.data.from_pandas(
        test_df, source=str(SPLITS_DIR / "test.parquet"), name="test"
    )

    all_results: list[dict] = []

    # ── Loop sobre variantes ─────────────────────────────────────────────── #
    for variant in args.variants:
        spec = VARIANTS[variant]
        logger.info(
            f"\n{'═' * 70}\nVariante {variant}: X={spec['X']} | Y={spec['Y']}\n{'═' * 70}"
        )

        # Construir X e Y para la variante
        t0 = time.perf_counter()
        X = build_x(variant, df_absa)
        Y = build_y(variant, df_absa)
        build_time = time.perf_counter() - t0
        logger.info(f"  X: {X.shape}  Y: {Y.shape}  build={build_time:.1f}s")

        x_diag = matrix_diagnostics(X, "X")
        y_diag = matrix_diagnostics(Y, "Y")
        logger.info(
            f"  X: mean={x_diag['X_mean']:.3f} row_var={x_diag['X_row_var_mean']:.3f} "
            f"sat_hi={x_diag['X_frac_saturated_high']:.3f}"
        )
        logger.info(
            f"  Y: mean={y_diag['Y_mean']:.3f} row_var={y_diag['Y_row_var_mean']:.3f} "
            f"sat_hi={y_diag['Y_frac_saturated_high']:.3f}"
        )

        models: dict[str, object] = {
            "CFAttention": CFAttention(
                k_neighbors=args.k_neighbors,
                min_sim=args.min_sim,
            ),
            "CBQuality": CBQuality(),
        }
        fit_data: dict[str, tuple] = {
            "CFAttention": (X, A),
            "CBQuality": (X, Y, A),
        }

        for name, model in models.items():
            run_name = f"{args.prefix}_{variant}_{name}".lstrip("_")
            with mlflow.start_run(run_name=run_name):
                mlflow.log_input(train_dataset, context="training")
                mlflow.log_input(test_dataset, context="testing")

                mlflow.set_tags(
                    {
                        "variant": variant,
                        "x_formula": spec["X"],
                        "y_formula": spec["Y"],
                        "model": name,
                    }
                )
                mlflow.log_params(
                    {
                        "variant": variant,
                        "x_fn": spec["x_fn"],
                        "y_fn": spec["y_fn"],
                        "A_shape": str(A.shape),
                        "X_shape": str(X.shape),
                        "Y_shape": str(Y.shape),
                        "n_test_users": n_test_users,
                        **model.get_params(),
                    }
                )
                mlflow.log_metrics({**x_diag, **y_diag})

                logger.info(f"Entrenando {name} (variante {variant})…")
                t0 = time.perf_counter()
                model.fit(*fit_data[name])
                fit_time = time.perf_counter() - t0
                mlflow.log_metric("fit_time_sec", fit_time)

                for k in args.k:
                    t0 = time.perf_counter()
                    metrics = evaluate_model(model, ground_truth, k=k, Y=Y)
                    eval_time = time.perf_counter() - t0

                    logger.info(
                        f"  [{variant}] {name:<14} K={k:<3} "
                        f"Hit={metrics['hit']:.4f} P={metrics['precision']:.4f} "
                        f"R={metrics['recall']:.4f} NDCG={metrics['ndcg']:.4f} "
                        f"MRR={metrics['mrr']:.4f} ILD={metrics['ild']:.4f} "
                        f"Cov={metrics['coverage']:.4f}"
                    )

                    mlflow.log_metrics(
                        {
                            "hit": metrics["hit"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "ndcg": metrics["ndcg"],
                            "mrr": metrics["mrr"],
                            "ild": metrics["ild"],
                            "coverage": metrics["coverage"],
                            "eval_time_sec": eval_time,
                        },
                        step=k,
                    )

                    all_results.append(
                        {
                            "variant": variant,
                            "model": name,
                            "K": k,
                            **metrics,
                        }
                    )

                # Guardar X e Y como artifacts para reproducibilidad
                with tempfile.TemporaryDirectory() as tmpdir:
                    x_path = Path(tmpdir) / f"X_{variant}.parquet"
                    y_path = Path(tmpdir) / f"Y_{variant}.parquet"
                    X.to_parquet(x_path)
                    Y.to_parquet(y_path)
                    mlflow.log_artifact(str(x_path))
                    mlflow.log_artifact(str(y_path))

    # ── Tabla resumen ────────────────────────────────────────────────────── #
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("RESULTADOS DE ABLACIÓN — X / Y")
    print("=" * 80)

    for k in args.k:
        sub = results_df[results_df["K"] == k].drop(columns="K")
        print(f"\n── K = {k} ──")
        pivot = sub.pivot_table(
            index=["variant", "model"],
            values=["hit", "precision", "recall", "ndcg", "mrr", "ild", "coverage"],
        )
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output, index=False)
        logger.info(f"\nResultados guardados en {args.output}")


if __name__ == "__main__":
    main()
