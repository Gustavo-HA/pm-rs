"""
Script para ejecutar el pipeline de extracción de aspectos y sentimientos
sobre el dataset filtrado de REST-MEX.

Uso:
    uv run python scripts/run_aspects.py
    uv run python scripts/run_aspects.py --splitter char --batch-size 128
    uv run python scripts/run_aspects.py --device cpu --output data/rest-mex/custom_output.parquet
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

import sys

sys.path.append(str(Path(__file__).parent.parent))  # Para importar mtrs

from mtrs.aspects import (
    AspectSentimentExtractor,
    split_review_punkt,
    split_review_char,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/rest-mex/splits")
INPUT_FILE = DATA_DIR / "train.parquet"

SPLITTERS = {
    "punkt": split_review_punkt,
    "char": split_review_char,
}

OUTPUT_NAMES = {
    "punkt": DATA_DIR / "absa" / "punkt_aspect_sentiment.parquet",
    "char": DATA_DIR / "absa" / "char_aspect_sentiment.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline ABSA: extracción de aspectos y sentimiento."
    )
    parser.add_argument(
        "--splitter",
        choices=SPLITTERS.keys(),
        default="punkt",
        help="Estrategia de segmentación (default: punkt).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tamaño de lote para inferencia en GPU (default: 64).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Dispositivo de inferencia: 'cuda' o 'cpu' (default: cuda).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta de salida del parquet. Si no se especifica, se usa el default por splitter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path: Path = args.output or OUTPUT_NAMES[args.splitter]
    split_fn = SPLITTERS[args.splitter]

    logger.info(f"Dataset: {INPUT_FILE}")
    logger.info(f"Splitter: {args.splitter} | Batch size: {args.batch_size} | Device: {args.device}")
    logger.info(f"Output: {output_path}")

    df = pd.read_parquet(INPUT_FILE)
    logger.info(f"Registros cargados: {len(df):,}")

    extractor = AspectSentimentExtractor(device=args.device)
    result = extractor.extract_from_dataframe(df, split_fn, batch_size=args.batch_size)

    result.to_parquet(output_path, index=False)
    logger.info(f"Resultado guardado en {output_path} ({len(result):,} filas)")


if __name__ == "__main__":
    main()
