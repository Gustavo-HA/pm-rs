"""
Distribución de las recomendaciones sobre el catálogo y rendimiento por
segmento de actividad.

Parte 1 — Exposición:
  Con las listas top-10 de cada modelo (y del híbrido en alpha*(beta=0.5)):
  exposición por pueblo, curva de Lorenz + coeficiente de Gini, relación
  exposición-popularidad (Spearman) y pueblos nunca recomendados.

Parte 2 — Ganador por usuario:
  Comparación pareada de NDCG@10 entre CBQuality y CFAttention, cruzada con
  el tamaño del historial y el cluster temático del perfil.

Parte 3 — Segmentos de actividad:
  NDCG@10 y Cov@10 por modelo dentro de segmentos por número de pueblos en
  el historial de entrenamiento.

Requiere: analyze_hybrid_sweep.py y analyze_profiles.py ejecutados antes.

Uso:
    uv run python scripts/analyze_recommendations.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from analysis_common import (
    FIG_DIR,
    MODEL_COLORS,
    RESULTS_DIR,
    load_matrices,
    setup_style,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS = ["CBQuality", "CFAttention", "CFClassic", "Hibrido"]
LABELS = {
    "CBQuality": "CBQuality",
    "CFAttention": "CFAttention",
    "CFClassic": "CFClassic",
    "Hibrido": "Híbrido",
}
SEGMENT_BINS = [0, 1, 2, 4, np.inf]
SEGMENT_LABELS = ["1", "2", "3–4", "≥5"]


def gini(counts: np.ndarray) -> float:
    """Coeficiente de Gini sobre el vector de exposiciones (incluye ceros)."""
    x = np.sort(counts.astype(float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def load_topk(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / f"topk10_{name}.csv")


def main() -> None:
    setup_style()
    A, X, Y = load_matrices()
    pueblos = list(A.columns)
    n_users_test = None

    topk = {m: load_topk(m) for m in MODELS}
    per_user = {
        m: pd.read_csv(RESULTS_DIR / f"per_user_k10_{m}.csv", index_col="user")
        for m in MODELS
    }
    n_users_test = len(per_user["CBQuality"])

    # ── Parte 1: exposición por pueblo ───────────────────────────────────── #
    exposure = pd.DataFrame(index=pueblos)
    for m in MODELS:
        counts = topk[m]["pueblo"].value_counts()
        exposure[m] = counts.reindex(pueblos).fillna(0).astype(int)
    popularity = A.notna().sum(axis=0).rename("popularidad_train")
    exposure = exposure.join(popularity)
    exposure.index.name = "pueblo"
    exposure.to_csv(RESULTS_DIR / "recs_exposure.csv")

    ginis = {m: gini(exposure[m].values) for m in MODELS}
    spearman = {
        m: stats.spearmanr(exposure[m], exposure["popularidad_train"]).statistic
        for m in MODELS
    }
    never = {m: exposure.index[exposure[m] == 0].tolist() for m in MODELS}

    print("\nExposición por modelo (listas top-10 sobre usuarios de prueba):")
    for m in MODELS:
        print(
            f"  {LABELS[m]:<12} Gini={ginis[m]:.3f}  "
            f"Spearman(exposición, popularidad)={spearman[m]:.3f}  "
            f"pueblos nunca recomendados={len(never[m])}"
        )
        if never[m]:
            print(f"    → {', '.join(never[m])}")
    pd.DataFrame(
        {"gini": ginis, "spearman_popularidad": spearman,
         "n_nunca_recomendados": {m: len(never[m]) for m in MODELS}}
    ).to_csv(RESULTS_DIR / "recs_exposure_summary.csv")

    # Figura: curvas de Lorenz
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    for m in MODELS:
        x = np.sort(exposure[m].values.astype(float))
        lorenz = np.insert(np.cumsum(x) / x.sum(), 0, 0)
        frac = np.linspace(0, 1, len(lorenz))
        ax.plot(
            frac, lorenz, lw=1.9, color=MODEL_COLORS[LABELS[m]],
            label=f"{LABELS[m]} (Gini {ginis[m]:.2f})",
        )
    ax.plot([0, 1], [0, 1], color="#949494", lw=1.0, ls="--", label="Equidad perfecta")
    ax.set_xlabel("Fracción acumulada de pueblos")
    ax.set_ylabel("Fracción acumulada de exposición")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recs_lorenz.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figura: exposición vs popularidad (CBQuality y CFAttention)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
    for ax, m in zip(axes, ["CBQuality", "CFAttention"]):
        ax.scatter(
            exposure["popularidad_train"], exposure[m], s=26,
            color=MODEL_COLORS[LABELS[m]], alpha=0.75, lw=0,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Popularidad en entrenamiento (usuarios)")
        ax.set_title(
            rf"{LABELS[m]} ($\rho_s$ = {spearman[m]:.2f})", fontsize=11
        )
    axes[0].set_ylabel("Exposición (apariciones en top-10)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recs_exposure_popularity.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Parte 2: ganador por usuario (CBQuality vs CFAttention) ─────────── #
    ndcg = pd.DataFrame({m: per_user[m]["ndcg"] for m in MODELS})
    hist_size = A.notna().sum(axis=1).reindex(ndcg.index).rename("n_historial")
    clusters = pd.read_csv(
        RESULTS_DIR / "profiles_cluster_assignments.csv", index_col="user"
    )["cluster_tema"].reindex(ndcg.index)

    diff = ndcg["CBQuality"] - ndcg["CFAttention"]
    winner = pd.Series(
        np.where(diff > 0, "CBQuality", np.where(diff < 0, "CFAttention", "Empate")),
        index=ndcg.index,
        name="ganador",
    )
    segment = pd.cut(hist_size, bins=SEGMENT_BINS, labels=SEGMENT_LABELS)

    winner_df = pd.concat([winner, hist_size, segment.rename("segmento"), clusters], axis=1)
    winner_df.to_csv(RESULTS_DIR / "recs_winner_per_user.csv")

    overall = winner.value_counts(normalize=True)
    by_segment = (
        winner_df.groupby("segmento", observed=True)["ganador"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    by_cluster = (
        winner_df.groupby("cluster_tema")["ganador"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    print("\nGanador por usuario (NDCG@10, CBQuality vs CFAttention):")
    print(overall.round(3).to_string())
    print("\nPor segmento de historial:")
    print(by_segment.round(3).to_string())
    print("\nPor cluster temático:")
    print(by_cluster.round(3).to_string())
    by_segment.to_csv(RESULTS_DIR / "recs_winner_by_segment.csv")
    by_cluster.to_csv(RESULTS_DIR / "recs_winner_by_cluster.csv")

    # Solapamiento medio entre listas CB y CF
    cb_sets = topk["CBQuality"].groupby("user")["pueblo"].apply(set)
    cf_sets = topk["CFAttention"].groupby("user")["pueblo"].apply(set)
    jaccard = np.mean(
        [
            len(cb_sets[u] & cf_sets[u]) / len(cb_sets[u] | cf_sets[u])
            for u in ndcg.index
        ]
    )
    print(f"\nJaccard medio top-10 CBQuality vs CFAttention: {jaccard:.3f}")

    # ── Parte 3: métricas por segmento de actividad ──────────────────────── #
    seg_rows = []
    for m in MODELS:
        df_m = per_user[m].join(segment.rename("segmento"))
        tk_m = topk[m].merge(
            segment.rename("segmento"), left_on="user", right_index=True
        )
        for seg in SEGMENT_LABELS:
            sub = df_m[df_m["segmento"] == seg]
            cov = tk_m.loc[tk_m["segmento"] == seg, "pueblo"].nunique() / len(pueblos)
            seg_rows.append(
                {
                    "model": LABELS[m],
                    "segmento": seg,
                    "n_usuarios": len(sub),
                    "ndcg": sub["ndcg"].mean(),
                    "hit": sub["hit"].mean(),
                    "coverage": cov,
                }
            )
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(RESULTS_DIR / "recs_segments.csv", index=False)
    print("\nMétricas por segmento:")
    print(
        seg_df.pivot(index="segmento", columns="model", values="ndcg")
        .round(3).to_string()
    )

    # Figura: NDCG y cobertura por segmento
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    xpos = np.arange(len(SEGMENT_LABELS))
    width = 0.19
    for i, m in enumerate(MODELS):
        sub = seg_df[seg_df["model"] == LABELS[m]].set_index("segmento")
        for ax, metric in zip(axes, ["ndcg", "coverage"]):
            ax.bar(
                xpos + (i - 1.5) * width,
                sub.loc[SEGMENT_LABELS, metric],
                width * 0.92,
                color=MODEL_COLORS[LABELS[m]],
                label=LABELS[m] if metric == "ndcg" else None,
            )
    axes[0].set_ylabel("NDCG@10")
    axes[1].set_ylabel("Cov@10 del segmento")
    for ax in axes:
        ax.set_xticks(xpos)
        ax.set_xticklabels(SEGMENT_LABELS)
        ax.set_xlabel("Pueblos en el historial de entrenamiento")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recs_segments.pdf", bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Listo — {n_users_test:,} usuarios de prueba analizados.")


if __name__ == "__main__":
    main()
