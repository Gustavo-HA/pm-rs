"""
Generate publication-ready charts for avance_abril evaluation metrics.

Reads mrr.csv, hit.csv, ndcg.csv, coverage.csv from reports/avance_abril/
and writes one PNG + PDF per metric to the same folder.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent / "reports" / "avance_abril"

METRIC_FILES = {
    "mrr":      BASE_DIR / "mrr.csv",
    "hit":      BASE_DIR / "hit.csv",
    "ndcg":     BASE_DIR / "ndcg.csv",
    "coverage": BASE_DIR / "coverage.csv",
}

METRIC_LABELS = {
    "mrr":      "MRR (Mean Reciprocal Rank)",
    "hit":      "Hit Rate",
    "ndcg":     "NDCG (Normalized Discounted Cumulative Gain)",
    "coverage": "Cobertura del Catálogo",
}

METRIC_YLABELS = {
    "mrr":      "MRR@K",
    "hit":      "Hit Rate@K",
    "ndcg":     "NDCG@K",
    "coverage": "Cobertura@K",
}

# ---------------------------------------------------------------------------
# Model name mapping (run name → display name)
# ---------------------------------------------------------------------------
NAME_MAP = {
    "_HybridFusion":    "Híbrido",
    "_CBQuality":       "CB Calidad",
    "_CFAttention":     "CF Atención",
    "_CFClassic":       "CF Clásico",
}

# Display order (most important first for legend readability)
MODEL_ORDER = [
    "Híbrido",
    "CB Calidad",
    "CF Atención",
    "CF Clásico",
]

# ColorBrewer Set1; paired linestyle/marker for grayscale safety
STYLES = {
    "Híbrido":          dict(color="#e41a1c", marker="o",  linestyle="-",   linewidth=2.5, markersize=7,  zorder=5),
    "CB Calidad":       dict(color="#377eb8", marker="s",  linestyle="-",   linewidth=2.0, markersize=6),
    "CF Atención":      dict(color="#4daf4a", marker="^",  linestyle="-",   linewidth=2.0, markersize=6),
    "CF Clásico":       dict(color="#984ea3", marker="v",  linestyle="--",  linewidth=2.0, markersize=6),
}

# ---------------------------------------------------------------------------
# Matplotlib global settings (LaTeX-ready fonts)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Georgia"],
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    10,
    "legend.title_fontsize": 11,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

K_VALUES = [1, 3, 5, 10, 20]


def load_metric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["model"] = df["Run"].map(NAME_MAP)
    df = df.dropna(subset=["model"])          # drop unmapped runs
    df = df[["model", "step", "value"]].copy()
    df = df.rename(columns={"step": "k"})
    df["k"] = df["k"].astype(int)
    return df


def plot_metric(metric: str, df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for model in MODEL_ORDER:
        subset = df[df["model"] == model].sort_values("k")
        if subset.empty:
            continue
        style = STYLES.get(model, {})
        ax.plot(
            subset["k"],
            subset["value"],
            label=model,
            **style,
        )

    ax.set_xlabel("K", fontweight="bold")
    ax.set_ylabel(METRIC_YLABELS[metric], fontweight="bold")
    ax.set_title(METRIC_LABELS[metric], fontweight="bold", pad=10)
    ax.set_xticks(K_VALUES)
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    # Y-axis: percentages for all metrics
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # ax.legend(
    #     title="Modelo",
    #     loc="upper left",
    #     framealpha=0.9,
    #     edgecolor="#cccccc",
    # )

    fig.tight_layout()

    stem = metric
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  ✓ {stem}.png / {stem}.pdf")


def main() -> None:
    print(f"Output dir: {BASE_DIR}")
    for metric, path in METRIC_FILES.items():
        df = load_metric(path)
        plot_metric(metric, df, BASE_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
