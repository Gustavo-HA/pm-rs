"""
Histogramas de la partición train/test (esquema last-month-out por usuario).

Genera dos figuras para el capítulo de metodología:

  1. split_test_fraction.pdf — eje x: fracción de los pueblos visitados por el
     usuario que quedaron en el conjunto de prueba; eje y: número de usuarios.
  2. split_test_count.pdf    — eje x: número de pueblos del usuario en prueba;
     eje y: número de usuarios.

Ambas figuras marcan la media y la mediana con líneas verticales etiquetadas.

Uso:
    uv run python scripts/plot_split_histograms.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLITS_DIR = Path("data/rest-mex/splits")
FIG_DIR = Path("reports/Tesis/figures")

C_BAR = "#4C72B0"
C_MEAN = "#C44E52"
C_MEDIAN = "#55A868"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    }
)


def load_per_user() -> pd.DataFrame:
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    test = pd.read_parquet(SPLITS_DIR / "test.parquet")

    n_train = train.groupby("Author")["Pueblo"].nunique().rename("n_train")
    n_test = test.groupby("Author")["Pueblo"].nunique().rename("n_test")
    df = pd.concat([n_train, n_test], axis=1).fillna(0).astype(int)
    df["n_total"] = df["n_train"] + df["n_test"]
    df = df[df["n_test"] > 0]  # usuarios con al menos un pueblo en prueba
    df["frac_test"] = df["n_test"] / df["n_total"]
    return df


def _vlines(ax, mean: float, median: float, fmt: str) -> None:
    ax.axvline(mean, color=C_MEAN, ls="--", lw=1.8, label=f"Media = {mean:{fmt}}")
    ax.axvline(
        median, color=C_MEDIAN, ls="-.", lw=1.8, label=f"Mediana = {median:{fmt}}"
    )
    ax.legend(frameon=False, fontsize=10)


def plot_fraction(df: pd.DataFrame, out: Path) -> None:
    vals = df["frac_test"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bins = np.arange(0.0, 1.0 + 0.05, 0.05)
    ax.hist(vals, bins=bins, color=C_BAR, edgecolor="white", linewidth=0.6)
    _vlines(ax, vals.mean(), float(np.median(vals)), ".3f")
    ax.set_xlabel("Fracción de pueblos del usuario asignados a prueba")
    ax.set_ylabel("Número de usuarios")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_count(df: pd.DataFrame, out: Path) -> None:
    vals = df["n_test"].to_numpy()
    vmax = int(vals.max())
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bins = np.arange(0.5, vmax + 1.5, 1.0)
    ax.hist(vals, bins=bins, color=C_BAR, edgecolor="white", linewidth=0.6)
    _vlines(ax, vals.mean(), float(np.median(vals)), ".2f")
    ax.set_xlabel("Número de pueblos del usuario asignados a prueba")
    ax.set_ylabel("Número de usuarios")
    ax.set_yscale("log")
    ax.set_xticks(range(1, vmax + 1))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_per_user()

    # Resumen para la redacción
    excluded = pd.read_parquet(SPLITS_DIR / "excluded.parquet")["Author"].nunique()
    print(f"usuarios train/test : {len(df):,}")
    print(f"usuarios excluidos  : {excluded:,}")
    print(
        "frac_test  -> media {:.3f}  mediana {:.3f}".format(
            df["frac_test"].mean(), df["frac_test"].median()
        )
    )
    print(
        "n_test     -> media {:.3f}  mediana {:.1f}  max {:d}".format(
            df["n_test"].mean(), df["n_test"].median(), int(df["n_test"].max())
        )
    )
    print(
        "1 - frac (proporción media en train) = {:.3f}".format(
            1 - df["frac_test"].mean()
        )
    )

    plot_fraction(df, FIG_DIR / "split_test_fraction.pdf")
    plot_count(df, FIG_DIR / "split_test_count.pdf")
    print(f"figuras escritas en {FIG_DIR}/")


if __name__ == "__main__":
    main()
