"""
Estudio de caso: recomendaciones explicables estilo EFM.

Selecciona un usuario representativo por cluster temático (hotel, cultura y
naturaleza, gastronomía) con historial suficiente y acierto en prueba, y
descompone el score de CBQuality de sus recomendaciones en contribuciones
por aspecto:

    r̂_{u,p} = Σ_a w_{u,a} · Y_{p,a},   w_{u,a} = X_{u,a} / Σ_a' X_{u,a'}

La contribución relativa de cada aspecto (w_{u,a}·Y_{p,a} / r̂) es la base
de la explicación "se recomienda p porque valoras a y p destaca en a".

Los usuarios se pseudonimizan (Usuario A, B, C) para el documento.

Requiere: analyze_hybrid_sweep.py y analyze_profiles.py ejecutados antes.

Uso:
    uv run python scripts/analyze_case_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from analysis_common import (
    FIG_DIR,
    RESULTS_DIR,
    build_ground_truth,
    load_matrices,
    load_splits,
    setup_style,
)

# Aspecto firma de cada cluster temático (ver profiles_cluster_summary_norm.csv)
CLUSTER_SIGNATURE = {0: "habitación", 1: "cultura", 2: "comida"}
PSEUDONYMS = {0: "Usuario A", 1: "Usuario B", 2: "Usuario C"}
MIN_HISTORY = 3
TOP_N = 5


def main() -> None:
    setup_style()
    A, X, Y = load_matrices()
    _, test = load_splits()
    gt = build_ground_truth(A, test)

    W = X.div(X.sum(axis=1), axis=0)
    clusters = pd.read_csv(
        RESULTS_DIR / "profiles_cluster_assignments.csv", index_col="user"
    )["cluster_tema"]
    per_user_cb = pd.read_csv(
        RESULTS_DIR / "per_user_k10_CBQuality.csv", index_col="user"
    )
    topk_cb = (
        pd.read_csv(RESULTS_DIR / "topk10_CBQuality.csv")
        .groupby("user")["pueblo"].apply(list)
    )
    hist_size = A.notna().sum(axis=1)

    # ── Selección determinista: mayor peso en el aspecto firma del cluster ── #
    chosen: dict[int, str] = {}
    for c, aspect in CLUSTER_SIGNATURE.items():
        candidates = clusters[clusters == c].index
        candidates = [
            u for u in candidates
            if hist_size.get(u, 0) >= MIN_HISTORY
            and u in per_user_cb.index
            and per_user_cb.loc[u, "hit"] == 1.0
        ]
        ranked = W.loc[candidates, aspect].sort_values(ascending=False)
        chosen[c] = ranked.index[0]

    # ── Figura: perfiles X de los tres usuarios ──────────────────────────── #
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4), sharey=True)
    aspects = list(X.columns)
    for ax, (c, user) in zip(axes, chosen.items()):
        vals = X.loc[user, aspects]
        sig = CLUSTER_SIGNATURE[c]
        colors = ["#0173B2" if a == sig else "#A8C6E8" for a in aspects]
        ax.bar(range(len(aspects)), vals, color=colors, width=0.72)
        ax.set_xticks(range(len(aspects)))
        ax.set_xticklabels(aspects, rotation=45, ha="right", fontsize=8.5)
        ax.set_title(PSEUDONYMS[c], fontsize=11)
        ax.set_ylim(1, 5.15)
    axes[0].set_ylabel("Importancia $X_{u,a}$")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case_profiles.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Detalle por usuario ─────────────────────────────────────────────── #
    records = []
    for c, user in chosen.items():
        hist = list(A.loc[user].dropna().index)
        future = sorted(gt[user])
        recs = topk_cb[user][:TOP_N]
        print(f"\n===== {PSEUDONYMS[c]} (cluster {c}: {CLUSTER_SIGNATURE[c]}) =====")
        print(f"  historial ({len(hist)}): {', '.join(hist)}")
        print(f"  prueba: {', '.join(future)}")
        print(f"  ndcg@10 CBQuality: {per_user_cb.loc[user, 'ndcg']:.3f}")
        w = W.loc[user]
        for rank, p in enumerate(recs, 1):
            contrib = (w * Y.loc[p]).sort_values(ascending=False)
            score = contrib.sum()
            share = contrib / score
            top3 = ", ".join(f"{a} ({share[a]:.0%})" for a in contrib.index[:3])
            is_hit = "HIT" if p in gt[user] else ""
            print(f"  {rank}. {p:<28} score={score:.3f}  {top3}  {is_hit}")
            records.append(
                {
                    "pseudonimo": PSEUDONYMS[c],
                    "cluster": c,
                    "rank": rank,
                    "pueblo": p,
                    "score": score,
                    "hit": p in gt[user],
                    **{f"contrib_{a}": share[a] for a in aspects},
                }
            )
    pd.DataFrame(records).to_csv(RESULTS_DIR / "case_study.csv", index=False)
    print(f"\nDetalle guardado en {RESULTS_DIR / 'case_study.csv'}")


if __name__ == "__main__":
    main()
