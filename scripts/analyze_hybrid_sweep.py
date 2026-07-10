"""
Regenera los resultados base del capítulo de resultados con las matrices
canónicas (variante D de la ablación, data/rest-mex/matrices/zs-bert-d):

  1. Tabla de modelos individuales (CBQuality, CFClassic, CFAttention)
     para K en {5, 10, 20} — CSV + filas LaTeX.
  2. Barrido de alpha a K=10 (51 puntos) — CSV.
  3. Frontera alpha*(beta) — CSV.
  4. Figuras alpha_sweep.pdf, alphaVSbeta.pdf, pareto_realized.pdf
     (mismo estilo que plot_hybrid_figures.py, del que importa las funciones).
  5. Métricas por usuario a K=10 y listas top-10 por modelo (insumo de los
     demás scripts de análisis).

La rama CFAttention se toma del stack serializado en models/hybrid_ab
(X y A no cambian entre variantes); la rama CBQuality se recomputa con la
Y de la variante D. CFClassic se reentrena (semilla fija).

Uso:
    uv run python scripts/analyze_hybrid_sweep.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from analysis_common import (
    FIG_DIR,
    RESULTS_DIR,
    branch_scores,
    build_ground_truth,
    catalog_coverage,
    cb_quality_scores,
    fused_scores,
    load_hybrid,
    load_matrices,
    load_splits,
    per_user_metrics,
    setup_style,
    topk_lists,
)
from mtrs.models import CFClassic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

K_VALUES = [5, 10, 20]
K_MAIN = 10
ALPHA_GRID = np.round(np.linspace(0.0, 1.0, 51), 4)
BETA_GRID = np.round(np.linspace(0.0, 1.0, 21), 4)


def cf_classic_scores(A: pd.DataFrame) -> pd.DataFrame:
    """Entrena CFClassic (defaults de run_models.py) y devuelve su matriz de scores."""
    model = CFClassic(n_factors=100, n_epochs=20, lr=0.01, reg=0.02, random_state=42)
    model.fit(A)
    raw = model._mu + model._bu[:, None] + model._bp[None, :] + model._P @ model._Q.T
    return pd.DataFrame(
        np.clip(raw, 1.0, 5.0), index=model._users, columns=model._pueblos
    )


def save_topk(topk: dict[str, list[str]], name: str) -> None:
    rows = [
        {"user": u, "rank": r + 1, "pueblo": p}
        for u, recs in topk.items()
        for r, p in enumerate(recs)
    ]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"topk{K_MAIN}_{name}.csv", index=False)


def main() -> None:
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    A, X, Y = load_matrices()
    _, test = load_splits()
    gt = build_ground_truth(A, test)
    users = list(gt.keys())
    n_pueblos = len(A.columns)
    logger.info(f"{len(users):,} usuarios de prueba, {n_pueblos} pueblos")

    hybrid = load_hybrid()
    s_cf, _ = branch_scores(hybrid)
    s_cb = cb_quality_scores(X, Y)
    logger.info("Entrenando CFClassic…")
    s_cfc = cf_classic_scores(A)

    model_scores = {"CBQuality": s_cb, "CFClassic": s_cfc, "CFAttention": s_cf}

    # ── 1. Tabla de modelos individuales ─────────────────────────────────── #
    rows = []
    for name, S in model_scores.items():
        for k in K_VALUES:
            tk = topk_lists(S, A, k=k, users=users)
            m = per_user_metrics(tk, gt, Y)
            cov = catalog_coverage(tk, n_pueblos)
            rows.append(
                {
                    "model": name,
                    "K": k,
                    "hit": m["hit"].mean(),
                    "precision": m["precision"].mean(),
                    "recall": m["recall"].mean(),
                    "ndcg": m["ndcg"].mean(),
                    "mrr": m["mrr"].mean(),
                    "ild": m["ild"].mean(),
                    "coverage": cov,
                }
            )
            if k == K_MAIN:
                m.to_csv(RESULTS_DIR / f"per_user_k{K_MAIN}_{name}.csv")
                save_topk(tk, name)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "model_eval.csv", index=False)

    print("\nFilas LaTeX (tab:modelos):")
    for name in model_scores:
        sub = table[table["model"] == name]
        for i, (_, r) in enumerate(sub.iterrows()):
            label = name if i == 0 else ""
            print(
                f"\t\t{label:<15} & {int(r['K']):<3} & {r['precision']:.3f} & "
                f"{r['recall']:.3f} & {r['ndcg']:.3f} & {r['mrr']:.3f} & "
                f"{r['ild']:.3f} & {r['coverage']:.3f} \\\\"
            )

    # ── 2. Barrido de alpha a K=10 ───────────────────────────────────────── #
    logger.info("Barrido de alpha…")
    sweep = []
    for alpha in ALPHA_GRID:
        S = fused_scores(s_cf, s_cb, float(alpha))
        tk = topk_lists(S, A, k=K_MAIN, users=users)
        m = per_user_metrics(tk, gt, Y)
        sweep.append(
            {
                "alpha": alpha,
                "ndcg": m["ndcg"].mean(),
                "coverage": catalog_coverage(tk, n_pueblos),
                "hit": m["hit"].mean(),
                "recall": m["recall"].mean(),
                "ild": m["ild"].mean(),
            }
        )
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(RESULTS_DIR / f"alpha_sweep_k{K_MAIN}.csv", index=False)

    # ── 3. Frontera alpha*(beta) ─────────────────────────────────────────── #
    ndcg = sweep_df["ndcg"].values
    cov = sweep_df["coverage"].values
    frontier = []
    for beta in BETA_GRID:
        f = beta * ndcg + (1 - beta) * cov
        i = int(f.argmax())
        frontier.append(
            {
                "beta": beta,
                "alpha_star": sweep_df["alpha"].iloc[i],
                "f_star": f[i],
                "ndcg_star": ndcg[i],
                "coverage_star": cov[i],
            }
        )
    frontier_df = pd.DataFrame(frontier)
    frontier_df.to_csv(RESULTS_DIR / "alpha_star_frontier.csv", index=False)

    # Híbrido en el compromiso beta=0.5 (insumo de significancia y casos)
    a_mid = float(frontier_df.loc[frontier_df["beta"] == 0.5, "alpha_star"].iloc[0])
    S_mid = fused_scores(s_cf, s_cb, a_mid)
    tk_mid = topk_lists(S_mid, A, k=K_MAIN, users=users)
    m_mid = per_user_metrics(tk_mid, gt, Y)
    m_mid.to_csv(RESULTS_DIR / f"per_user_k{K_MAIN}_Hibrido.csv")
    save_topk(tk_mid, "Hibrido")
    with open(RESULTS_DIR / "alpha_star_beta05.txt", "w") as fh:
        fh.write(f"{a_mid}\n")
    logger.info(f"alpha*(beta=0.5) = {a_mid}")

    # ── 4. Figuras ───────────────────────────────────────────────────────── #
    from plot_hybrid_figures import plot_alpha_sweep, plot_alpha_vs_beta, plot_pareto

    alpha_arr = sweep_df["alpha"].values
    plot_alpha_sweep(alpha_arr, ndcg, cov, FIG_DIR / "alpha_sweep.pdf")
    plot_alpha_vs_beta(alpha_arr, ndcg, cov, FIG_DIR / "alphaVSbeta.pdf")
    plot_pareto(alpha_arr, ndcg, cov, FIG_DIR / "pareto_realized.pdf")
    logger.info(f"Figuras regeneradas en {FIG_DIR}/")

    # Resumen para la redacción
    sat = cov >= cov.max() - 1e-9
    print(f"\nNDCG@10: {ndcg[0]:.4f} (alpha=0) → {ndcg[-1]:.4f} (alpha=1)")
    print(f"Cov@10:  {cov[0]:.4f} (alpha=0) → {cov[-1]:.4f} (alpha=1)")
    print(f"Cov@10 se satura a partir de alpha = {alpha_arr[sat][0]:.2f}")
    print(frontier_df.to_string(index=False))


if __name__ == "__main__":
    main()
