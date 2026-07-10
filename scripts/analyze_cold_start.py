"""
Simulación del perfil por encuesta para usuarios nuevos (arranque en frío).

Un usuario nuevo no tiene reseñas de las que extraer X_u, pero puede
responder una encuesta de 8 preguntas ("del 1 al 5, ¿cuánto te importa cada
aspecto?") cuya respuesta vive en la misma escala que X. La simulación
cuantiza el perfil continuo de los usuarios de prueba a los niveles que
capturaría la encuesta y mide cuánta calidad de recomendación se pierde:

  - Encuesta de 5 niveles: X' = round(X)  →  {1, 2, 3, 4, 5}
  - Encuesta de 3 niveles: X' ∈ {1, 3, 5} por terciles del rango

Se re-puntúan CBQuality (matching X'·Y), CFAttention (vecindad sobre X') y
el híbrido en alpha*(beta=0.5), manteniendo el protocolo de evaluación.
Además se reporta la estabilidad del ranking de CBQuality (Spearman por
usuario y solapamiento del top-10) entre el perfil continuo y el cuantizado.

Requiere: analyze_hybrid_sweep.py ejecutado antes (alpha*).

Uso:
    uv run python scripts/analyze_cold_start.py
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
    build_ground_truth,
    catalog_coverage,
    cb_quality_scores,
    cf_attention_scores,
    fused_scores,
    load_matrices,
    load_splits,
    per_user_metrics,
    setup_style,
    topk_lists,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

K = 10


def quantize_5(X: pd.DataFrame) -> pd.DataFrame:
    return X.round().clip(1, 5)


def quantize_3(X: pd.DataFrame) -> pd.DataFrame:
    edges = [1.0, 1 + 4 / 3, 1 + 8 / 3, 5.0]
    levels = np.select(
        [X.values <= edges[1], X.values <= edges[2]], [1.0, 3.0], default=5.0
    )
    return pd.DataFrame(levels, index=X.index, columns=X.columns)


def main() -> None:
    setup_style()
    A, X, Y = load_matrices()
    _, test = load_splits()
    gt = build_ground_truth(A, test)
    users = list(gt.keys())
    n_pueblos = len(A.columns)

    alpha_star = float(
        (RESULTS_DIR / "alpha_star_beta05.txt").read_text().strip()
    )
    logger.info(f"alpha*(beta=0.5) = {alpha_star}")

    variants = {
        "Perfil continuo (ABSA)": X,
        "Encuesta 5 niveles": quantize_5(X),
        "Encuesta 3 niveles": quantize_3(X),
    }

    rows = []
    s_cb_cont = None
    tk_cb_cont = None
    for source, Xv in variants.items():
        logger.info(f"Evaluando: {source}")
        s_cb = cb_quality_scores(Xv, Y)
        s_cf = cf_attention_scores(Xv, A, k_neighbors=20)
        model_scores = {
            "CBQuality": s_cb,
            "CFAttention": s_cf,
            "Híbrido": fused_scores(s_cf, s_cb, alpha_star),
        }
        for name, S in model_scores.items():
            tk = topk_lists(S, A, k=K, users=users)
            m = per_user_metrics(tk, gt, Y)
            rows.append(
                {
                    "perfil": source,
                    "model": name,
                    "ndcg": m["ndcg"].mean(),
                    "hit": m["hit"].mean(),
                    "coverage": catalog_coverage(tk, n_pueblos),
                }
            )
        if source == "Perfil continuo (ABSA)":
            s_cb_cont = s_cb
            tk_cb_cont = topk_lists(s_cb, A, k=K, users=users)
        else:
            # Estabilidad del ranking de CBQuality frente al perfil continuo
            rho = np.mean(
                [
                    stats.spearmanr(s_cb_cont.loc[u], s_cb.loc[u]).statistic
                    for u in users
                ]
            )
            tk_cb = topk_lists(s_cb, A, k=K, users=users)
            overlap = np.mean(
                [
                    len(set(tk_cb_cont[u]) & set(tk_cb[u])) / K
                    for u in users
                ]
            )
            logger.info(
                f"  CBQuality vs continuo: Spearman medio={rho:.3f}  "
                f"solapamiento top-10={overlap:.3f}"
            )
            rows[-3]["spearman_vs_continuo"] = rho
            rows[-3]["solapamiento_top10"] = overlap

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "cold_start.csv", index=False)
    print("\nResultados de la simulación de encuesta:")
    print(df.round(4).to_string(index=False))

    # ── Figura: NDCG y cobertura por fuente de perfil ────────────────────── #
    sources = list(variants.keys())
    models = ["CBQuality", "CFAttention", "Híbrido"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    xpos = np.arange(len(sources))
    width = 0.24
    for i, name in enumerate(models):
        sub = df[df["model"] == name].set_index("perfil").loc[sources]
        for ax, metric in zip(axes, ["ndcg", "coverage"]):
            ax.bar(
                xpos + (i - 1) * width,
                sub[metric],
                width * 0.9,
                color=MODEL_COLORS[name],
                label=name if metric == "ndcg" else None,
            )
    short = ["Continuo\n(ABSA)", "Encuesta\n5 niveles", "Encuesta\n3 niveles"]
    axes[0].set_ylabel("NDCG@10")
    axes[1].set_ylabel("Cov@10")
    for ax in axes:
        ax.set_xticks(xpos)
        ax.set_xticklabels(short, fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cold_start.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Listo.")


if __name__ == "__main__":
    main()
