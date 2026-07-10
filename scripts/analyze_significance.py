"""
Pruebas de significancia estadística sobre las diferencias de NDCG@10.

Compara por pares, con la prueba de rangos con signo de Wilcoxon pareada por
usuario (zero_method='pratt', aproximación normal), los cuatro modelos:
CBQuality, CFAttention, CFClassic y el híbrido en alpha*(beta=0.5).
Los p-valores se corrigen por comparaciones múltiples con el método de Holm
y se reporta el tamaño de efecto r = |z| / sqrt(n).

Requiere: analyze_hybrid_sweep.py ejecutado antes (métricas por usuario).

Uso:
    uv run python scripts/analyze_significance.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).parent))

from analysis_common import RESULTS_DIR

MODELS = ["CBQuality", "Hibrido", "CFClassic", "CFAttention"]
LABELS = {
    "CBQuality": "CBQuality",
    "CFAttention": "CFAttention",
    "CFClassic": "CFClassic",
    "Hibrido": "Híbrido ($\\alpha^{\\star}$)",
}


def holm(pvals: list[float]) -> list[float]:
    """Corrección de Holm-Bonferroni (step-down)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def main() -> None:
    ndcg = pd.DataFrame(
        {
            m: pd.read_csv(RESULTS_DIR / f"per_user_k10_{m}.csv", index_col="user")[
                "ndcg"
            ]
            for m in MODELS
        }
    ).dropna()
    n = len(ndcg)

    rows = []
    for a, b in combinations(MODELS, 2):
        res = stats.wilcoxon(
            ndcg[a], ndcg[b], zero_method="pratt", method="approx"
        )
        diff = ndcg[a] - ndcg[b]
        rows.append(
            {
                "modelo_a": a,
                "modelo_b": b,
                "media_a": ndcg[a].mean(),
                "media_b": ndcg[b].mean(),
                "delta_medias": diff.mean(),
                "z": res.zstatistic,
                "p": res.pvalue,
                "efecto_r": abs(res.zstatistic) / np.sqrt(n),
                "n": n,
            }
        )
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p"].tolist())
    df.to_csv(RESULTS_DIR / "significance_ndcg10.csv", index=False)

    print(f"Wilcoxon pareado por usuario sobre NDCG@10 (n={n:,}):\n")
    print(df.round(4).to_string(index=False))

    print("\nFilas LaTeX:")
    for _, r in df.iterrows():
        p_str = "$<10^{-4}$" if r["p_holm"] < 1e-4 else f"{r['p_holm']:.3f}"
        print(
            f"\t\t{LABELS[r['modelo_a']]} & {LABELS[r['modelo_b']]} & "
            f"{r['delta_medias']:+.3f} & {p_str} & {r['efecto_r']:.3f} \\\\"
        )


if __name__ == "__main__":
    main()
