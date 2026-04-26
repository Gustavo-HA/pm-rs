"""
Etapa de optimización de pesos del modelo híbrido de fusión lineal.

Flujo:
  1. Carga matrices A, X, Y, R y el split de test.
  2. Ajusta los cuatro modelos base (CFClassic, CFMultiCriteria, CBQuality, CBAttention).
  3. Construye HybridFusion y precomputa el score_stack.
  4. Ejecuta los tres optimizadores (PSO, DE, BO) sobre la métrica objetivo.
  5. Evalúa los pesos óptimos con el protocolo completo (evaluate_model).
  6. Registra todos los experimentos en MLflow.
  7. Imprime tabla comparativa de resultados.

Uso:
    uv run scripts/run_optimization.py
    uv run scripts/run_optimization.py --metric ndcg --k 10
    uv run scripts/run_optimization.py --metric mrr --k 5 10 20 --pso-iter 200
    uv run scripts/run_optimization.py --optimizers pso de  # sólo PSO y DE
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

from mtrs.models import CBAttention, CBQuality, CFClassic, CFMultiCriteria, HybridFusion
from mtrs.optimization import BayesianOptimizer, DEOptimizer, OptimizationResult, PSOOptimizer

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
# Métricas completas (consistentes con run_models.py)                         #
# --------------------------------------------------------------------------- #


def hit_at_k(recs, relevant):
    return 1.0 if any(r in relevant for r in recs) else 0.0


def precision_at_k(recs, relevant):
    return sum(1 for r in recs if r in relevant) / len(recs) if recs else 0.0


def recall_at_k(recs, relevant):
    return sum(1 for r in recs if r in relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs, relevant):
    dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(recs) if r in relevant)
    ideal = min(len(recs), len(relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(recs, relevant):
    for i, r in enumerate(recs):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def intra_list_diversity(recs, Y):
    available = [p for p in recs if p in Y.index]
    if len(available) < 2:
        return 0.0
    vecs = Y.loc[available].values
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms == 0, 1.0, norms)
    sim = vecs @ vecs.T
    triu = sim[np.triu_indices(len(available), k=1)]
    return float(1.0 - triu.mean())


def evaluate_model(model, ground_truth, k, Y):
    per_user = []
    all_recs: set[str] = set()
    for user, relevant in ground_truth.items():
        recs = [p for p, _ in model.recommend(user, k=k)]
        all_recs.update(recs)
        per_user.append({
            "hit": hit_at_k(recs, relevant),
            "precision": precision_at_k(recs, relevant),
            "recall": recall_at_k(recs, relevant),
            "ndcg": ndcg_at_k(recs, relevant),
            "mrr": mrr(recs, relevant),
            "ild": intra_list_diversity(recs, Y),
        })
    agg = {m: float(np.mean([u[m] for u in per_user])) for m in per_user[0]}
    agg["coverage"] = len(all_recs) / len(model._all_pueblos) if model._all_pueblos else 0.0
    return agg


# --------------------------------------------------------------------------- #
# Artifacts                                                                    #
# --------------------------------------------------------------------------- #


def _log_optimization_artifacts(result: OptimizationResult, run_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = Path(tmpdir) / f"{run_name}_weights.npz"
        np.savez(
            weights_path,
            weights=result.best_weights,
            history=np.array(result.history),
        )
        mlflow.log_artifact(str(weights_path))


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description="Optimización de pesos del modelo híbrido.")
    p.add_argument("--matrices-dir", type=Path, default=MATRICES_DIR)
    p.add_argument(
        "--metric",
        type=str,
        default="ndcg",
        choices=["ndcg", "mrr", "hit", "precision", "recall", "ild", "coverage"],
        help="Métrica objetivo para la optimización (default: ndcg).",
    )
    p.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[10],
        help="Valores de K para top-K (default: 10).",
    )
    p.add_argument(
        "--opt-k",
        type=int,
        default=None,
        help="K usado durante la optimización. Default: primer valor de --k.",
    )
    p.add_argument(
        "--optimizers",
        nargs="+",
        default=["pso", "de", "bayesian"],
        choices=["pso", "de", "bayesian"],
        help="Optimizadores a ejecutar (default: todos).",
    )
    # PSO
    p.add_argument("--pso-particles", type=int, default=30)
    p.add_argument("--pso-iter", type=int, default=150)
    # DE
    p.add_argument("--de-maxiter", type=int, default=300)
    p.add_argument("--de-popsize", type=int, default=15)
    # BO
    p.add_argument("--bo-initial", type=int, default=12)
    p.add_argument("--bo-iter", type=int, default=50)
    p.add_argument("--bo-xi", type=float, default=0.01)
    # Modelos base
    p.add_argument("--cf-factors", type=int, default=100)
    p.add_argument("--cf-epochs", type=int, default=20)
    p.add_argument("--cf-lr", type=float, default=0.005)
    p.add_argument("--cf-reg", type=float, default=0.02)
    p.add_argument("--k-neighbors", type=int, default=20)
    # MLflow
    p.add_argument("--experiment", type=str, default="pm-rs-optimization")
    p.add_argument("--prefix", type=str, default="")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main():
    args = parse_args()
    matrices_dir = args.matrices_dir

    # ── Cargar matrices ──────────────────────────────────────────────────── #
    logger.info("Cargando matrices…")
    A = pd.read_parquet(matrices_dir / "A.parquet")
    X = pd.read_parquet(matrices_dir / "X.parquet")
    Y = pd.read_parquet(matrices_dir / "Y.parquet")
    R = pd.read_parquet(matrices_dir / "R.parquet")
    logger.info(f"  A:{A.shape}  X:{X.shape}  Y:{Y.shape}  R:{R.shape}")

    # ── Ground truth ─────────────────────────────────────────────────────── #
    test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")
    test_df = test_df[test_df["Author"].isin(A.index)]
    ground_truth: dict[str, set[str]] = (
        test_df.groupby("Author")["Pueblo"].apply(set).to_dict()
    )
    logger.info(f"  {len(ground_truth):,} usuarios de test")

    # ── K de optimización ────────────────────────────────────────────────── #
    opt_k = args.opt_k if args.opt_k is not None else args.k[0]
    logger.info(f"  Optimizando sobre {args.metric}@{opt_k}")

    # ── Ajustar modelos base ─────────────────────────────────────────────── #
    logger.info("Ajustando modelos base…")
    base_models = {
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
    fit_data = {
        "CFClassic": (A,),
        "CFMultiCriteria": (R, A),
        "CBAttention": (X, A),
        "CBQuality": (X, Y, A),
    }
    for name, model in base_models.items():
        t0 = time.perf_counter()
        model.fit(*fit_data[name])
        logger.info(f"  {name} fit: {time.perf_counter()-t0:.1f}s")

    # ── HybridFusion — precomputa score_stack ────────────────────────────── #
    logger.info("Construyendo HybridFusion y precomputando score_stack…")
    hybrid = HybridFusion(base_models)
    t0 = time.perf_counter()
    hybrid.fit(A)
    logger.info(f"  HybridFusion fit: {time.perf_counter()-t0:.1f}s")

    # ── Definir optimizadores ────────────────────────────────────────────── #
    optimizer_registry: dict[str, object] = {
        "pso": PSOOptimizer(
            n_particles=args.pso_particles,
            n_iter=args.pso_iter,
            random_state=42,
        ),
        "de": DEOptimizer(
            maxiter=args.de_maxiter,
            popsize=args.de_popsize,
            random_state=42,
        ),
        "bayesian": BayesianOptimizer(
            n_initial=args.bo_initial,
            n_iter=args.bo_iter,
            xi=args.bo_xi,
            random_state=42,
        ),
    }

    # ── MLflow ───────────────────────────────────────────────────────────── #
    mlflow.set_tracking_uri("http://0.0.0.0:1825")
    mlflow.set_experiment(args.experiment)

    dataset_params = {
        "A_shape": str(A.shape),
        "X_shape": str(X.shape),
        "Y_shape": str(Y.shape),
        "R_shape": str(R.shape),
        "n_test_users": len(ground_truth),
        "opt_metric": args.metric,
        "opt_k": opt_k,
    }

    all_results: list[dict] = []

    # ── Baseline: pesos iguales ──────────────────────────────────────────── #
    hybrid.set_weights(np.ones(len(base_models)) / len(base_models))
    run_name = f"{args.prefix}_HybridEqual" if args.prefix else "HybridEqual"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(dataset_params)
        mlflow.log_params({"optimizer": "equal_weights"})
        mlflow.log_param("weights", ",".join(f"{w:.4f}" for w in hybrid.weights))
        for k in args.k:
            metrics = evaluate_model(hybrid, ground_truth, k, Y)
            mlflow.log_metrics(metrics, step=k)
            all_results.append({"model": "HybridEqual", "K": k, **metrics})
            logger.info(
                f"  HybridEqual  NDCG@{k}={metrics['ndcg']:.4f}  "
                f"MRR={metrics['mrr']:.4f}  Cov={metrics['coverage']:.4f}"
            )

    # ── Ejecutar cada optimizador ────────────────────────────────────────── #
    opt_results: dict[str, OptimizationResult] = {}

    for opt_name in args.optimizers:
        optimizer = optimizer_registry[opt_name]
        display_name = type(optimizer).__name__

        logger.info(f"\n{'='*60}")
        logger.info(f"Ejecutando {display_name}…")
        t0 = time.perf_counter()

        result = optimizer.optimize(
            hybrid=hybrid,
            ground_truth=ground_truth,
            Y=Y,
            k=opt_k,
            metric=args.metric,
        )
        opt_time = time.perf_counter() - t0
        opt_results[opt_name] = result

        logger.info(
            f"{display_name} completado en {opt_time:.1f}s  "
            f"best_{args.metric}@{opt_k}={result.best_score:.4f}  "
            f"evals={result.n_evals}"
        )

        # ── Evaluar con pesos óptimos ──────────────────────────────────── #
        hybrid.set_weights(result.best_weights)

        run_name = f"{args.prefix}_{display_name}" if args.prefix else display_name
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(dataset_params)
            mlflow.log_params({
                "optimizer": display_name,
                "opt_metric": args.metric,
                "opt_k": opt_k,
                "opt_time_sec": round(opt_time, 2),
                "n_evals": result.n_evals,
                "weights": ",".join(f"{w:.4f}" for w in result.best_weights),
            })

            for k in args.k:
                metrics = evaluate_model(hybrid, ground_truth, k, Y)
                mlflow.log_metrics(metrics, step=k)
                all_results.append({"model": display_name, "K": k, **metrics})
                logger.info(
                    f"  {display_name:<20} "
                    f"NDCG@{k}={metrics['ndcg']:.4f}  "
                    f"MRR={metrics['mrr']:.4f}  "
                    f"Cov={metrics['coverage']:.4f}"
                )

            _log_optimization_artifacts(result, display_name)

    # ── Tabla resumen ────────────────────────────────────────────────────── #
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print(f"OPTIMIZACIÓN DE PESOS — métrica objetivo: {args.metric}@{opt_k}")
    print("=" * 80)

    for k in args.k:
        sub = results_df[results_df["K"] == k].set_index("model").drop(columns="K")
        sub.columns = [
            f"{c}@{k}" if c not in ("ild", "coverage") else c for c in sub.columns
        ]
        print(f"\n── K = {k} ──")
        print(sub.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n── Pesos óptimos por optimizador ──")
    model_names = list(base_models.keys())
    header = f"{'Optimizer':<20}" + "".join(f"  {n:<16}" for n in model_names)
    print(header)
    print("-" * len(header))
    for opt_name, result in opt_results.items():
        row = f"{result.optimizer_name:<20}"
        row += "".join(f"  {w:<16.4f}" for w in result.best_weights)
        print(row)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output, index=False)
        logger.info(f"\nResultados guardados en {args.output}")


if __name__ == "__main__":
    main()
