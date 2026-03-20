"""
Genera los splits temporales train/test del filtered_dataset a nivel pueblo-usuario.

Estrategia: per-user, last-month-out
  - Para cada (Author, Pueblo) se toma la primera fecha de visita (FechaEstadia.min()).
  - El "último mes" del usuario es el mes-año de su visita máxima.
  - Todos los pueblos cuya primera visita cae en ese último mes → test.
  - Los pueblos restantes → train.
  - Usuarios donde el 100% de sus pueblos caen en test → excluidos de ambos splits.

Salida (en data/rest-mex/splits/):
  - train.parquet  — reviews de pueblos en el split train
  - test.parquet   — reviews de pueblos en el split test
  - excluded.parquet — reviews de usuarios excluidos (para auditoría)

Uso:
    uv run python scripts/split_dataset.py
    uv run python scripts/split_dataset.py --input data/rest-mex/processed/filtered_dataset.parquet
    uv run python scripts/split_dataset.py --output-dir data/rest-mex/splits
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/rest-mex")
DEFAULT_INPUT = DATA_DIR / "processed" / "filtered_dataset.parquet"
DEFAULT_OUTPUT = DATA_DIR / "splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split temporal train/test per-user a nivel pueblo."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Parquet de entrada (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directorio de salida (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def compute_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retorna (train_reviews, test_reviews, excluded_reviews).

    La asignación se hace a nivel (Author, Pueblo): todas las reviews de un
    par (Author, Pueblo) van al mismo split, según la estrategia last-month-out.
    """
    # Primera visita de cada (Author, Pueblo)
    first_visit = (
        df.groupby(["Author", "Pueblo"])["FechaEstadia"]
        .min()
        .rename("FirstVisit")
        .reset_index()
    )

    # Último mes por usuario (año-mes de la visita más reciente)
    last_visit = first_visit.groupby("Author")["FirstVisit"].max().rename("LastVisit")
    first_visit = first_visit.join(last_visit, on="Author")

    # Pueblo en test si su primera visita es en el mismo mes-año que la última
    first_visit["in_test"] = (
        first_visit["FirstVisit"].dt.to_period("M")
        == first_visit["LastVisit"].dt.to_period("M")
    )

    # Usuarios con 100% pueblos en test → excluir
    user_stats = first_visit.groupby("Author")["in_test"].agg(["sum", "count"])
    excluded_authors = user_stats[user_stats["sum"] == user_stats["count"]].index

    logger.info(
        f"Usuarios excluidos (100% pueblos en test): {len(excluded_authors):,} "
        f"({100 * len(excluded_authors) / user_stats.shape[0]:.1f}%)"
    )

    first_visit_filtered = first_visit[~first_visit["Author"].isin(excluded_authors)]

    test_pairs = first_visit_filtered.loc[
        first_visit_filtered["in_test"], ["Author", "Pueblo"]
    ]
    train_pairs = first_visit_filtered.loc[
        ~first_visit_filtered["in_test"], ["Author", "Pueblo"]
    ]

    # Merge con reviews originales
    df_valid = df[~df["Author"].isin(excluded_authors)]
    df_excluded = df[df["Author"].isin(excluded_authors)]

    train = df_valid.merge(train_pairs, on=["Author", "Pueblo"])
    test = df_valid.merge(test_pairs, on=["Author", "Pueblo"])

    return train, test, df_excluded


def log_split_stats(train: pd.DataFrame, test: pd.DataFrame, excluded: pd.DataFrame) -> None:
    total_reviews = len(train) + len(test) + len(excluded)
    logger.info("—— Resumen del split ——")
    logger.info(f"  Reviews totales (original): {total_reviews:,}")
    logger.info(
        f"  Train:    {len(train):,} reviews | "
        f"{train['Author'].nunique():,} usuarios | "
        f"{train['Pueblo'].nunique()} pueblos"
    )
    logger.info(
        f"  Test:     {len(test):,} reviews | "
        f"{test['Author'].nunique():,} usuarios | "
        f"{test['Pueblo'].nunique()} pueblos"
    )
    logger.info(
        f"  Excluidos: {len(excluded):,} reviews | "
        f"{excluded['Author'].nunique():,} usuarios"
    )

    # Distribución de % test por usuario
    user_total = (
        pd.concat([train[["Author", "Pueblo"]], test[["Author", "Pueblo"]]])
        .drop_duplicates()
        .groupby("Author")
        .size()
        .rename("total")
    )
    user_test = (
        test[["Author", "Pueblo"]]
        .drop_duplicates()
        .groupby("Author")
        .size()
        .rename("test")
    )
    pct = (user_test / user_total).dropna()
    logger.info(
        f"  Test% por usuario — mean={pct.mean():.1%}  "
        f"median={pct.median():.1%}  "
        f"p25={pct.quantile(0.25):.1%}  "
        f"p75={pct.quantile(0.75):.1%}"
    )


def main() -> None:
    args = parse_args()

    logger.info(f"Input:      {args.input}")
    logger.info(f"Output dir: {args.output_dir}")

    logger.info("Cargando dataset…")
    df = pd.read_parquet(args.input)
    logger.info(f"  {len(df):,} reviews | {df['Author'].nunique():,} usuarios | {df['Pueblo'].nunique()} pueblos")

    logger.info("Calculando split temporal (last-month-out)…")
    train, test, excluded = compute_splits(df)

    log_split_stats(train, test, excluded)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.output_dir / "train.parquet", index=False)
    test.to_parquet(args.output_dir / "test.parquet", index=False)
    excluded.to_parquet(args.output_dir / "excluded.parquet", index=False)

    logger.info(f"Splits guardados en {args.output_dir}/")


if __name__ == "__main__":
    main()
