"""
Análisis de la canalización ABSA para el capítulo de metodología.

Genera:
  1. absa_confidence.pdf      — distribución de la confianza del clasificador de
     aspecto (zero-shot NLI) y de sentimiento (RoBERTa) sobre la salida final.
  2. absa_aspect_by_tipo.pdf  — composición de aspectos por tipo de establecimiento
     (muestra que los aspectos están condicionados por el Tipo).

Además imprime el comparativo Punkt vs Char (selección del segmentador).

Uso:
    uv run python scripts/plot_absa_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ABSA_DIR = Path("data/rest-mex/absa")
FIG_DIR = Path("reports/Tesis/figures")

PUNKT = ABSA_DIR / "punkt_bert_absa.parquet"  # pipeline final (BERT-NLI + RoBERTa)
CHAR = ABSA_DIR / "char_aspect_sentiment.parquet"

C_ASPECT = "#4C72B0"
C_SENT = "#55A868"
C_MED = "#C44E52"

# orden de aspectos por columna (los 8 aspectos del proyecto)
ASPECT_ORDER = [
    "servicio",
    "precio",
    "ambiente",
    "comida",
    "habitación",
    "ubicación",
    "naturaleza",
    "cultura",
]
TIPO_ORDER = ["Hotel", "Restaurant", "Attractive"]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    }
)


def splitter_comparison() -> None:
    p = pd.read_parquet(PUNKT)
    c = pd.read_parquet(CHAR)

    def stats(df: pd.DataFrame) -> dict:
        frags = df.drop_duplicates(subset=["review_idx", "sentence"])
        return {
            "fragmentos": frags.shape[0],
            "frags/reseña": frags.shape[0] / df["review_idx"].nunique(),
            "long_mediana": frags["sentence"].str.len().median(),
            "conf_aspecto": df["aspect_score"].median(),
            "conf_sentim": df["sentiment_score"].median(),
        }

    sp, sc = stats(p), stats(c)
    print("\n== Comparativo Punkt vs Char ==")
    print(f"{'métrica':16s} {'Punkt':>12s} {'Char':>12s}")
    for k in sp:
        print(f"{k:16s} {sp[k]:>12.3f} {sc[k]:>12.3f}")


def plot_confidence() -> None:
    df = pd.read_parquet(PUNKT)
    asp = df["aspect_score"].to_numpy()
    sen = df["sentiment_score"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    for ax, vals, color, title in (
        (axes[0], asp, C_ASPECT, "Confianza de aspecto (zero-shot NLI)"),
        (axes[1], sen, C_SENT, "Confianza de sentimiento (RoBERTa)"),
    ):
        ax.hist(vals, bins=50, range=(0, 1), color=color, edgecolor="white", linewidth=0.3)
        med = float(np.median(vals))
        ax.axvline(med, color=C_MED, ls="--", lw=1.8, label=f"Mediana = {med:.3f}")
        ax.set_xlabel("Confianza")
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=10)
    axes[0].set_ylabel("Número de fragmentos")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "absa_confidence.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_aspect_by_tipo() -> None:
    df = pd.read_parquet(PUNKT)
    ct = pd.crosstab(df["Tipo"], df["aspect"], normalize="index") * 100
    aspects = [a for a in ASPECT_ORDER if a in ct.columns]
    ct = ct.reindex(index=TIPO_ORDER, columns=aspects)

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    cmap = plt.get_cmap("tab10")
    left = np.zeros(len(TIPO_ORDER))
    y = np.arange(len(TIPO_ORDER))
    for j, a in enumerate(aspects):
        vals = ct[a].to_numpy()
        ax.barh(y, vals, left=left, color=cmap(j % 10), edgecolor="white", label=a)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v >= 6:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white")
        left += np.nan_to_num(vals)
    ax.set_yticks(y)
    ax.set_yticklabels(TIPO_ORDER)
    ax.set_xlabel("Porcentaje de fragmentos (%)")
    ax.set_xlim(0, 100)
    ax.legend(ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.22), loc="upper center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "absa_aspect_by_tipo.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    splitter_comparison()
    plot_confidence()
    plot_aspect_by_tipo()
    print(f"\nfiguras escritas en {FIG_DIR}/")


if __name__ == "__main__":
    main()
