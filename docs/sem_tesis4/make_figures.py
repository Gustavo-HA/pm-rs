"""Genera las figuras para la presentacion Seminario de Tesis 4.

Lee las metricas del barrido de alpha desde MLflow (run del experimento
pm-rs-alpha) y construye `alpha_metrics.pdf`: dos paneles NDCG@10 vs alpha y
Cov@10 vs alpha, anotando el alpha* maximo de NDCG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mlflow.tracking import MlflowClient

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

TRACKING_URI = "http://localhost:1825"
RUN_ID = "19a676103f484bee98e5c01dc9eeb02b"
STEP_SCALE = 1000

CREAM = "#F2EEE3"
NAVY = "#2F3E4E"
ACCENT_NDCG = "#2E7D32"
ACCENT_COV = "#7E57C2"
ACCENT_STAR = "#C62828"

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.facecolor": CREAM,
        "figure.facecolor": CREAM,
        "savefig.facecolor": CREAM,
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
    }
)


def fetch_metric(client: MlflowClient, run_id: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    history = client.get_metric_history(run_id=run_id, key=key)
    history = sorted(history, key=lambda h: h.step)
    alphas = np.array([h.step for h in history], dtype=float) / STEP_SCALE
    values = np.array([h.value for h in history], dtype=float)
    return alphas, values


def main() -> None:
    client = MlflowClient(TRACKING_URI)
    alpha, ndcg = fetch_metric(client, RUN_ID, "ndcg_at_10")
    _, cov = fetch_metric(client, RUN_ID, "coverage_at_10")

    idx_star_ndcg = int(np.argmax(ndcg))
    alpha_star_ndcg = float(alpha[idx_star_ndcg])
    ndcg_star = float(ndcg[idx_star_ndcg])
    cov_at_star = float(cov[idx_star_ndcg])

    idx_star_cov = int(np.argmax(cov))
    alpha_star_cov = float(alpha[idx_star_cov])
    cov_star = float(cov[idx_star_cov])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)

    ax = axes[0]
    ax.plot(alpha, ndcg, color=ACCENT_NDCG, linewidth=2.4, marker="o", markersize=4.5)
    ax.axvline(alpha_star_ndcg, color=ACCENT_STAR, linestyle="--", linewidth=1.4, alpha=0.85)
    ax.scatter([alpha_star_ndcg], [ndcg_star], color=ACCENT_STAR, s=90, zorder=5,
               edgecolors="white", linewidths=1.2)
    ax.annotate(
        rf"$\alpha^{{\star}}_{{\mathrm{{NDCG}}}}={alpha_star_ndcg:.2f}$"
        + "\n"
        + rf"NDCG$={ndcg_star:.3f}$",
        xy=(alpha_star_ndcg, ndcg_star),
        xytext=(0.55, 0.20),
        textcoords="axes fraction",
        fontsize=11,
        color=NAVY,
        arrowprops=dict(arrowstyle="->", color=ACCENT_STAR, lw=1.2),
    )
    ax.set_xlabel(r"$\alpha$  (peso de CBAttention)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Ranking", color=NAVY)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1)

    ax = axes[1]
    ax.plot(alpha, cov, color=ACCENT_COV, linewidth=2.4, marker="o", markersize=4.5)
    ax.axvline(alpha_star_ndcg, color=ACCENT_STAR, linestyle="--", linewidth=1.4, alpha=0.85,
               label=rf"$\alpha^{{\star}}_{{\mathrm{{NDCG}}}}={alpha_star_ndcg:.2f}$")
    ax.scatter([alpha_star_ndcg], [cov_at_star], color=ACCENT_STAR, s=90, zorder=5,
               edgecolors="white", linewidths=1.2)
    ax.annotate(
        rf"Cov$={cov_at_star:.3f}$",
        xy=(alpha_star_ndcg, cov_at_star),
        xytext=(0.18, 0.78),
        textcoords="axes fraction",
        fontsize=11,
        color=NAVY,
        arrowprops=dict(arrowstyle="->", color=ACCENT_STAR, lw=1.2),
    )
    ax.axhline(cov[0], color=NAVY, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(0.02, cov[0] + 0.005, f"CBQuality puro ({cov[0]:.2f})",
            fontsize=9, color=NAVY, alpha=0.85)
    ax.axhline(cov[-1], color=NAVY, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(0.02, cov[-1] - 0.025, f"CBAttention puro ({cov[-1]:.2f})",
            fontsize=9, color=NAVY, alpha=0.85)
    ax.set_xlabel(r"$\alpha$  (peso de CBAttention)")
    ax.set_ylabel("Coverage@10")
    ax.set_title("Cobertura del catalogo", color=NAVY)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.suptitle(
        r"Barrido de $\alpha$: NDCG@10 y Coverage@10 desacopladas",
        fontsize=14, color=NAVY, y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "alpha_metrics.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura generada: {out}")
    print(f"  alpha*_NDCG = {alpha_star_ndcg:.4f}  NDCG@10 = {ndcg_star:.4f}  Cov@10 = {cov_at_star:.4f}")
    print(f"  alpha*_Cov  = {alpha_star_cov:.4f}  Cov@10 = {cov_star:.4f}")
    print(f"  NDCG@10(alpha=0) = {ndcg[0]:.4f}  NDCG@10(alpha=1) = {ndcg[-1]:.4f}")
    print(f"  Cov@10(alpha=0)  = {cov[0]:.4f}   Cov@10(alpha=1)  = {cov[-1]:.4f}")


if __name__ == "__main__":
    main()
