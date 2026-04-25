"""
Inferencia: genera top-K recomendaciones para un usuario específico.

Uso:
    uv run scripts/infer.py --user "NombreUsuario" --model CFClassic
    uv run scripts/infer.py --user "NombreUsuario" --model HybridFusion --k 10
    uv run scripts/infer.py --user "NombreUsuario" --model CBAttention --k 5 --include-visited
    uv run scripts/infer.py --list-users
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from mtrs.models import CBAttention, CBQuality, CFClassic, CFMultiCriteria, HybridFusion

DATA_DIR = Path("data/rest-mex")
MATRICES_DIR = DATA_DIR / "matrices" / "zs-bert-tempmedian"

MODELS = ["CFClassic", "CFMultiCriteria", "CBQuality", "CBAttention", "HybridFusion"]

# Hyperparameter defaults (match run_models.py)
CF_FACTORS = 100
CF_EPOCHS = 20
CF_LR = 0.005
CF_REG = 0.02
K_NEIGHBORS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia de recomendaciones.")
    parser.add_argument("--user", type=str, help="Identificador del usuario (Author).")
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        default="HybridFusion",
        help=f"Modelo a usar (default: HybridFusion). Opciones: {', '.join(MODELS)}.",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Número de recomendaciones (default: 10)."
    )
    parser.add_argument(
        "--include-visited",
        action="store_true",
        help="Incluir pueblos ya visitados por el usuario.",
    )
    parser.add_argument(
        "--matrices-dir",
        type=Path,
        default=MATRICES_DIR,
        help=f"Directorio con matrices (default: {MATRICES_DIR}).",
    )
    parser.add_argument(
        "--list-users",
        action="store_true",
        help="Listar todos los usuarios disponibles y salir.",
    )
    # Model hyperparameters
    parser.add_argument("--cf-factors", type=int, default=CF_FACTORS)
    parser.add_argument("--cf-epochs", type=int, default=CF_EPOCHS)
    parser.add_argument("--cf-lr", type=float, default=CF_LR)
    parser.add_argument("--cf-reg", type=float, default=CF_REG)
    parser.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS)
    return parser.parse_args()


def load_matrices(matrices_dir: Path) -> tuple[pd.DataFrame, ...]:
    A = pd.read_parquet(matrices_dir / "A.parquet")
    X = pd.read_parquet(matrices_dir / "X.parquet")
    Y = pd.read_parquet(matrices_dir / "Y.parquet")
    R = pd.read_parquet(matrices_dir / "R.parquet")
    return A, X, Y, R


def build_and_fit(
    model_name: str, A: pd.DataFrame, X: pd.DataFrame, Y: pd.DataFrame, R: pd.DataFrame, args: argparse.Namespace
) -> object:
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

    if model_name == "HybridFusion":
        print("Entrenando modelos base para HybridFusion…")
        for name, model in base_models.items():
            print(f"  → {name}…", end=" ", flush=True)
            model.fit(*fit_data[name])
            print("listo")
        hybrid = HybridFusion(base_models)
        hybrid.fit(A)
        return hybrid

    model = base_models[model_name]
    print(f"Entrenando {model_name}…", end=" ", flush=True)
    model.fit(*fit_data[model_name])
    print("listo")
    return model


def main() -> None:
    args = parse_args()
    matrices_dir = args.matrices_dir

    print(f"Cargando matrices desde {matrices_dir}…")
    A, X, Y, R = load_matrices(matrices_dir)

    if args.list_users:
        print(f"\n{len(A.index):,} usuarios disponibles:\n")
        for user in sorted(A.index):
            n_pueblos = A.loc[user].notna().sum()
            print(f"  {user}  ({n_pueblos} pueblos visitados)")
        return

    if not args.user:
        print("Error: especifica --user o usa --list-users para ver usuarios disponibles.")
        sys.exit(1)

    if args.user not in A.index:
        print(f"Error: usuario '{args.user}' no encontrado en la matriz A.")
        print("Usa --list-users para ver los usuarios disponibles.")
        sys.exit(1)

    # Load test ground truth if available
    test_path = DATA_DIR / "splits" / "test.parquet"
    relevant: set[str] = set()
    if test_path.exists():
        test_df = pd.read_parquet(test_path)
        user_test = test_df[test_df["Author"] == args.user]
        relevant = set(user_test["Pueblo"].unique())

    model = build_and_fit(args.model, A, X, Y, R, args)

    exclude_visited = not args.include_visited
    recs = model.recommend(args.user, k=args.k, exclude_visited=exclude_visited)

    visited = set(A.loc[args.user].dropna().index)
    n_visited = A.loc[args.user].notna().sum()

    print(f"\n{'=' * 60}")
    print(f"Usuario      : {args.user}")
    print(f"Modelo       : {args.model}")
    print(f"Pueblos visitados (train): {n_visited}")
    if relevant:
        print(f"Pueblos en test (ground truth): {len(relevant)}  → {', '.join(sorted(relevant))}")
    print(f"Top-{args.k} recomendaciones ({'incluye visitados' if args.include_visited else 'solo no visitados'}):")
    print(f"{'=' * 60}")
    print(f"{'Rank':<6} {'Pueblo':<35} {'Score':>8}  {'':>4}")
    print(f"{'-' * 60}")
    for rank, (pueblo, score) in enumerate(recs, start=1):
        tags = []
        if pueblo in relevant:
            tags.append("✓")
        if pueblo in visited:
            tags.append("*")
        tag_str = " " + "".join(tags) if tags else ""
        print(f"{rank:<6} {pueblo:<35} {score:>8.4f}{tag_str}")
    print(f"{'=' * 60}")
    scores = [score for _, score in recs]
    print(f"  Score  min={min(scores):.4f}  max={max(scores):.4f}  range={max(scores)-min(scores):.4f}")
    legend = []
    if relevant:
        legend.append("✓ = en test set (hit)")
    if args.include_visited:
        legend.append("* = visitado en train")
    if legend:
        print("  " + "  |  ".join(legend))


if __name__ == "__main__":
    main()
