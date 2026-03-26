"""
Entrena y evalúa los cuatro modelos del RecSys sobre el split test temporal.

Matrices de entrada (data/rest-mex/matrices/):
  A.parquet — ratings u×p
  X.parquet — atención usuario-aspecto
  Y.parquet — calidad pueblo-aspecto
  R.parquet — sentimiento u-p-j (MultiIndex Author×Pueblo)

Split de evaluación (data/rest-mex/splits/):
  test.parquet — pueblos en el "último mes" de cada usuario (ground truth)

Métricas calculadas @ K:
  Precision@K, Recall@K, NDCG@K, MRR, Coverage, Intra-list Diversity

Uso:
    uv run python scripts/run_models.py
    uv run python scripts/run_models.py --k 5 10 20
    uv run python scripts/run_models.py --output results/model_eval.csv
    uv run python scripts/run_models.py --experiment my-experiment
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
from scipy.spatial.distance import cdist

sys.path.append(str(Path(__file__).parent.parent))

from mtrs.models import CBAttention, CBQuality, CFClassic, CFMultiCriteria

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/rest-mex")
MATRICES_DIR = DATA_DIR / "matrices"
SPLITS_DIR = DATA_DIR / "splits"


# --------------------------------------------------------------------------- #
# Métricas                                                                     #
# --------------------------------------------------------------------------- #

def hit_at_k(recs: list[str], relevant: set[str]) -> float:
    return 1.0 if any(r in relevant for r in recs) else 0.0

def precision_at_k(recs: list[str], relevant: set[str]) -> float:
    if not recs:
        return 0.0
    hits = sum(1 for r in recs if r in relevant)
    return hits / len(recs)


def recall_at_k(recs: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in recs if r in relevant)
    return hits / len(relevant)


def ndcg_at_k(recs: list[str], relevant: set[str]) -> float:
    """Binary relevance NDCG@K."""
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, r in enumerate(recs)
        if r in relevant
    )
    ideal_hits = min(len(recs), len(relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(recs: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first hit, 0 if no hit."""
    for i, r in enumerate(recs):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def intra_list_diversity(recs: list[str], Y: pd.DataFrame) -> float:
    """Mean pairwise cosine distance between recommended pueblos in Y-space."""
    available = [p for p in recs if p in Y.index]
    if len(available) < 2:
        return 0.0
    vecs = Y.loc[available].values
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs_norm = vecs / norms
    sim_matrix = vecs_norm @ vecs_norm.T
    n = len(available)
    # Average pairwise distance (upper triangle)
    triu = sim_matrix[np.triu_indices(n, k=1)]
    return float(1.0 - triu.mean())


def evaluate_model(
    model,
    test_ground_truth: dict[str, set[str]],
    k: int,
    Y: pd.DataFrame,
) -> dict[str, float]:
    """
    Itera sobre todos los usuarios en test_ground_truth, genera top-K
    recomendaciones y computa métricas agregadas (macro-average).
    """
    metrics_per_user: list[dict[str, float]] = []
    all_recommended: set[str] = set()

    for user, relevant in test_ground_truth.items():
        recs = [pueblo for pueblo, _ in model.recommend(user, k=k)]
        all_recommended.update(recs)

        metrics_per_user.append({
            "hit": hit_at_k(recs, relevant),
            "precision": precision_at_k(recs, relevant),
            "recall": recall_at_k(recs, relevant),
            "ndcg": ndcg_at_k(recs, relevant),
            "mrr": mrr(recs, relevant),
            "ild": intra_list_diversity(recs, Y),
        })

    agg = {
        metric: float(np.mean([m[metric] for m in metrics_per_user]))
        for metric in metrics_per_user[0]
    }
    n_pueblos = len(model._all_pueblos)
    agg["coverage"] = len(all_recommended) / n_pueblos if n_pueblos else 0.0
    return agg


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación de modelos del RecSys.")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[10],
        help="Valores de K para top-K (default: 10).",
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
        default="pm-rs",
        help="Nombre del experimento en MLflow (default: pm-rs).",
    )
    parser.add_argument(
        "--cf-factors", type=int, default=100, help="Factores latentes para CFClassic."
    )
    parser.add_argument(
        "--cf-epochs", type=int, default=20, help="Épocas SGD para CFClassic."
    )
    parser.add_argument(
        "--k-neighbors", type=int, default=20, help="Vecinos para CFMultiCriteria / CBAttention."
    )
    parser.add_argument(
        "--cf-lr", type=float, default=0.005, help="Learning rate para CFClassic."
    )
    parser.add_argument(
        "--cf-reg", type=float, default=0.02, help="Regularización L2 lambda para CFClassic."
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Artifacts                                                                    #
# --------------------------------------------------------------------------- #

def _log_model_artifacts(name: str, model: object) -> None:
    """Guarda los pesos aprendidos del modelo como artifact en MLflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / f"{name}.npz"
        if name == "CFClassic":
            np.savez(
                artifact_path,
                mu=model._mu, bu=model._bu, bp=model._bp,
                P=model._P, Q=model._Q,
            )
        elif name == "CFMultiCriteria":
            np.savez(artifact_path, R_tensor=model._R_tensor)
        elif name == "CBAttention":
            np.savez(artifact_path, sim_matrix=model._sim_matrix, mu=model._mu)
        elif name == "CBQuality":
            np.savez(artifact_path, scores=model._scores.values)
        mlflow.log_artifact(str(artifact_path))


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    # ── Cargar matrices ──────────────────────────────────────────────────── #
    logger.info("Cargando matrices…")
    A = pd.read_parquet(MATRICES_DIR / "A.parquet")
    X = pd.read_parquet(MATRICES_DIR / "X.parquet")
    Y = pd.read_parquet(MATRICES_DIR / "Y.parquet")
    R = pd.read_parquet(MATRICES_DIR / "R.parquet")
    logger.info(f"  A: {A.shape}  X: {X.shape}  Y: {Y.shape}  R: {R.shape}")

    # ── Construir ground truth desde test ────────────────────────────────── #
    logger.info("Construyendo ground truth desde splits/test.parquet…")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")

    # Solo usuarios presentes en A (train)
    test_df = test_df[test_df["Author"].isin(A.index)]

    ground_truth: dict[str, set[str]] = (
        test_df.groupby("Author")["Pueblo"]
        .apply(set)
        .to_dict()
    )
    n_test_users = len(ground_truth)
    logger.info(f"  {n_test_users:,} usuarios de test con pueblos en A")

    # ── Registrar datasets para MLflow ───────────────────────────────────── #
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

    # ── Definir modelos ──────────────────────────────────────────────────── #
    models: dict[str, object] = {
        "CFClassic": CFClassic(
            n_factors=args.cf_factors,
            n_epochs=args.cf_epochs,
            lr=args.cf_lr,
            reg=args.cf_reg,
            random_state=42,
        ),
        "CFMultiCriteria": CFMultiCriteria(k_neighbors=args.k_neighbors),
        "CBAttention": CBAttention(k_neighbors=args.k_neighbors),
        "CBQuality": CBQuality(),
    }

    # ── Fit de datos para los modelos que requieren entrenamiento ─────── #
    fit_data: dict[str, tuple] = {
        "CFClassic": (A,),
        "CFMultiCriteria": (R, A),
        "CBAttention": (X, A),
        "CBQuality": (X, Y, A),
    }

    # ── MLflow — flat runs, uno por modelo ───────────────────────────────── #
    mlflow.set_tracking_uri("http://0.0.0.0:1825")
    mlflow.set_experiment(args.experiment)

    dataset_params = {
        "A_shape": str(A.shape),
        "X_shape": str(X.shape),
        "Y_shape": str(Y.shape),
        "R_shape": str(R.shape),
        "n_test_users": n_test_users,
    }

    all_results: list[dict] = []

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # ── Datasets ─────────────────────────────────────────────── #
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(test_dataset, context="testing")

            # ── Params: dataset metadata + model hyperparameters ──── #
            mlflow.log_params(dataset_params)
            mlflow.log_params(model.get_params())

            # ── Fit ──────────────────────────────────────────────────── #
            logger.info(f"Entrenando {name}…")
            t0 = time.perf_counter()
            model.fit(*fit_data[name])
            fit_time = time.perf_counter() - t0
            logger.info(f"  {name} fit: {fit_time:.1f}s")

            mlflow.log_metric("fit_time_sec", fit_time)

            # ── Eval para cada K ─────────────────────────────────────── #
            for k in args.k:
                t0 = time.perf_counter()
                metrics = evaluate_model(model, ground_truth, k=k, Y=Y)
                eval_time = time.perf_counter() - t0

                logger.info(
                    f"  {name:<20} "
                    f"Hit@{k}={metrics['hit']:.4f}  "
                    f"P@{k}={metrics['precision']:.4f}  "
                    f"R@{k}={metrics['recall']:.4f}  "
                    f"NDCG@{k}={metrics['ndcg']:.4f}  "
                    f"MRR={metrics['mrr']:.4f}  "
                    f"ILD={metrics['ild']:.4f}  "
                    f"Cov={metrics['coverage']:.4f}  "
                    f"({eval_time:.1f}s)"
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

                all_results.append({"model": name, "K": k, **metrics})

            # ── Artifacts: pesos del modelo ──────────────────────────── #
            _log_model_artifacts(name, model)

    # ── Tabla resumen ────────────────────────────────────────────────────── #
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("RESULTADOS DE EVALUACIÓN")
    print("=" * 80)

    for k in args.k:
        sub = results_df[results_df["K"] == k].set_index("model").drop(columns="K")
        sub.columns = [f"{c}@{k}" if c not in ("ild", "coverage") else c for c in sub.columns]
        print(f"\n── K = {k} ──")
        print(sub.to_string(float_format=lambda x: f"{x:.4f}"))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output, index=False)
        logger.info(f"\nResultados guardados en {args.output}")


if __name__ == "__main__":
    main()
