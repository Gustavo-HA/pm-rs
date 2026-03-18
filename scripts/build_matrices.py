"""
Script para construir las cuatro matrices del sistema de recomendación multi-criterio.

Entrada:
  - data/rest-mex/punkt_aspect_sentiment.parquet  (ABSA output → M1, M2, M3, M4)

Salida (en data/rest-mex/matrices/):
  - A.parquet  — Matriz de ratings        (m × n)
  - X.parquet  — Importancia usuario-aspecto (m × p)
  - Y.parquet  — Calidad pueblo-aspecto   (n × p)
  - R.parquet  — Sentimiento u-p-j        (m·n × p, MultiIndex Author×Pueblo)

Uso:
    uv run python scripts/build_matrices.py
    uv run python scripts/build_matrices.py --absa data/rest-mex/char_aspect_sentiment.parquet
    uv run python scripts/build_matrices.py --output-dir data/rest-mex/matrices_char
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from mtrs.aggregation.matrices import (
    compute_rating_matrix,
    compute_user_aspect_importance,
    compute_pueblo_aspect_quality,
    compute_user_pueblo_aspect_sentiment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/rest-mex")
DEFAULT_ABSA = DATA_DIR / "punkt_aspect_sentiment.parquet"
DEFAULT_OUTPUT = DATA_DIR / "matrices"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye las matrices M1–M4 del sistema de recomendación."
    )
    parser.add_argument(
        "--absa",
        type=Path,
        default=DEFAULT_ABSA,
        help=f"Parquet con salida ABSA (default: {DEFAULT_ABSA}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directorio de salida (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def _log_shape(name: str, df: pd.DataFrame) -> None:
    density = df.notna().mean().mean()
    logger.info(f"  {name}: shape={df.shape}  densidad={density:.3f}")


def main() -> None:
    args = parse_args()

    logger.info(f"ABSA input:  {args.absa}")
    logger.info(f"Output dir:  {args.output_dir}")

    logger.info("Cargando datos…")
    df_absa = pd.read_parquet(args.absa)
    logger.info(f"  ABSA: {len(df_absa):,} filas")

    logger.info("Construyendo matrices…")

    logger.info("M1 — Rating matrix A (u × p)")
    A = compute_rating_matrix(df_absa)
    _log_shape("A", A)

    logger.info("M2 — User-aspect importance X (u × j)")
    X = compute_user_aspect_importance(df_absa)
    _log_shape("X", X)

    logger.info("M3 — Pueblo-aspect quality Y (p × j)")
    Y = compute_pueblo_aspect_quality(df_absa)
    _log_shape("Y", Y)

    logger.info("M4 — User-pueblo-aspect sentiment R (u·p × j)")
    R = compute_user_pueblo_aspect_sentiment(df_absa)
    _log_shape("R", R)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    A.to_parquet(args.output_dir / "A.parquet")
    X.to_parquet(args.output_dir / "X.parquet")
    Y.to_parquet(args.output_dir / "Y.parquet")
    R.to_parquet(args.output_dir / "R.parquet")

    logger.info(f"Matrices guardadas en {args.output_dir}/")


if __name__ == "__main__":
    main()
