"""
Inferencia con el hibrido CFAttention + CBQuality usando alpha en runtime.

Carga el HybridFusion serializado (joblib), reasigna los pesos a [alpha, 1-alpha]
y genera top-K recomendaciones para el usuario indicado.

Uso:
    uv run python scripts/infer_hybrid.py --list-users
    uv run python scripts/infer_hybrid.py --user "NombreUsuario" --alpha 0.7
    uv run python scripts/infer_hybrid.py --user "NombreUsuario" --alpha 0.3 --k 5
    uv run python scripts/infer_hybrid.py --user "NombreUsuario" --alpha 1.0 --include-visited
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

DATA_DIR = Path("data/rest-mex")
DEFAULT_MODEL_PATH = Path("models/hybrid_ab/hybrid.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia del hibrido alpha-CFAttention/CBQuality."
    )
    parser.add_argument("--user", type=str, help="Identificador del usuario (Author).")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Peso del CFAttention. (1-alpha) va al CBQuality. (default: 0.5)",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Numero de recomendaciones (default: 10)."
    )
    parser.add_argument(
        "--include-visited",
        action="store_true",
        help="Incluir pueblos ya visitados por el usuario.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Ruta al joblib del hibrido (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--list-users",
        action="store_true",
        help="Listar usuarios disponibles en el hibrido y salir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        print(f"Error: no se encontro el hibrido en {args.model_path}.")
        print("Ejecuta primero: uv run python scripts/run_hybrid_alpha.py")
        sys.exit(1)

    if not 0.0 <= args.alpha <= 1.0:
        print(f"Error: alpha debe estar en [0, 1] (recibido {args.alpha}).")
        sys.exit(1)

    print(f"Cargando hibrido desde {args.model_path}…")
    hybrid = joblib.load(args.model_path)

    if args.list_users:
        users = sorted(hybrid._users)
        print(f"\n{len(users):,} usuarios disponibles:\n")
        for user in users:
            print(f"  {user}")
        return

    if not args.user:
        print("Error: especifica --user o usa --list-users.")
        sys.exit(1)

    if args.user not in hybrid._user_idx:
        print(f"Error: usuario '{args.user}' no encontrado en el hibrido.")
        print("Usa --list-users para ver los usuarios disponibles.")
        sys.exit(1)

    # Pesos: [alpha, 1-alpha] segun orden de modelos en el dict del hibrido.
    model_names = hybrid._model_names
    if model_names != ["CFAttention", "CBQuality"]:
        print(
            f"Advertencia: orden de modelos esperado ['CFAttention', 'CBQuality'], "
            f"encontrado {model_names}."
        )
    hybrid.set_weights(np.array([args.alpha, 1.0 - args.alpha]))

    # Ground truth opcional
    test_path = DATA_DIR / "splits" / "test.parquet"
    relevant: set[str] = set()
    if test_path.exists():
        test_df = pd.read_parquet(test_path)
        user_test = test_df[test_df["Author"] == args.user]
        relevant = set(user_test["Pueblo"].unique())

    exclude_visited = not args.include_visited
    recs = hybrid.recommend(args.user, k=args.k, exclude_visited=exclude_visited)

    visited = hybrid._visited(args.user)
    n_visited = len(visited)

    print(f"\n{'=' * 60}")
    print(f"Usuario      : {args.user}")
    print(f"Modelo       : HybridFusion (alpha={args.alpha:.4f})")
    print(f"  w_CFAttention = {args.alpha:.4f}")
    print(f"  w_CBQuality   = {1.0 - args.alpha:.4f}")
    print(f"Pueblos visitados (train): {n_visited}")
    if relevant:
        print(
            f"Pueblos en test (ground truth): {len(relevant)}  -> "
            f"{', '.join(sorted(relevant))}"
        )
    label = "incluye visitados" if args.include_visited else "solo no visitados"
    print(f"Top-{args.k} recomendaciones ({label}):")
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
    if recs:
        scores = [score for _, score in recs]
        print(
            f"  Score  min={min(scores):.4f}  max={max(scores):.4f}  "
            f"range={max(scores) - min(scores):.4f}"
        )
    legend = []
    if relevant:
        legend.append("✓ = en test set (hit)")
    if args.include_visited:
        legend.append("* = visitado en train")
    if legend:
        print("  " + "  |  ".join(legend))


if __name__ == "__main__":
    main()
