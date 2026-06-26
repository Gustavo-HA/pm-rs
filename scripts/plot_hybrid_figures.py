"""
Figuras del modelo híbrido para el capítulo de metodología.

Lee el barrido de alfa registrado en MLflow (experimento pm-rs-alpha-sweep,
tracking http://0.0.0.0:1825) y genera, todas a K=10:

  1. alpha_sweep.pdf     — NDCG@10 (relevancia) y Cov@10 (cobertura) vs alfa.
  2. alphaVSbeta.pdf     — trayectoria del óptimo alfa*(beta) y métricas en él
                           (análisis de sensibilidad, estilo del póster).
  3. pareto_realized.pdf — frente de Pareto realizado en el plano Cov@10–NDCG@10,
                           con los puntos no dominados resaltados.

Uso:
    uv run python scripts/plot_hybrid_figures.py
    uv run python scripts/plot_hybrid_figures.py --k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

FIG_DIR = Path("reports/Tesis/figures")
TRACKING_URI = "http://0.0.0.0:1825"
EXPERIMENT = "pm-rs-alpha-sweep"

C_NDCG = "#4C72B0"  # relevancia
C_COV = "#DD8452"  # cobertura
C_STAR = "#C44E52"  # óptimo / frente

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    }
)


# Barrido K=10 extraído de MLflow (pm-rs-alpha-sweep). Se embebe como respaldo
# para que las figuras sean reproducibles aunque el servidor de tracking esté apagado.
EMBEDDED_K10 = {
    "alpha": np.round(np.arange(0.0, 1.0001, 0.05), 2),
    "ndcg": np.array(
        [0.3118, 0.3053, 0.2975, 0.2867, 0.2735, 0.2598, 0.2470, 0.2352, 0.2246,
         0.2128, 0.2012, 0.1906, 0.1813, 0.1711, 0.1616, 0.1523, 0.1449, 0.1387,
         0.1324, 0.1267, 0.1217]
    ),
    "cov": np.array(
        [0.6458, 0.6667, 0.6875, 0.6875, 0.7083, 0.7500, 0.8125, 0.8333, 0.8750,
         0.8750, 0.8958, 0.9375, 0.9792, 0.9792, 0.9792, 0.9792, 1.0000, 1.0000,
         1.0000, 1.0000, 1.0000]
    ),
}


def load_sweep(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        client = MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT)
        if exp is None:
            raise RuntimeError(f"Experimento '{EXPERIMENT}' no encontrado")
        run = client.search_runs([exp.experiment_id], max_results=1)[0]
        rid = run.info.run_id

        def hist(metric: str) -> dict[int, float]:
            return {m.step: m.value for m in client.get_metric_history(rid, metric)}

        ndcg = hist(f"ndcg_at_{k}")
        cov = hist(f"coverage_at_{k}")
        steps = sorted(ndcg.keys())
        alpha = np.array([s / 1000.0 for s in steps])
        return alpha, np.array([ndcg[s] for s in steps]), np.array([cov[s] for s in steps])
    except Exception as exc:  # servidor apagado u otro fallo de red
        if k == 10:
            print(f"[aviso] MLflow no disponible ({exc}); uso datos embebidos K=10.")
            d = EMBEDDED_K10
            return d["alpha"], d["ndcg"], d["cov"]
        raise SystemExit(f"MLflow no disponible y no hay respaldo para K={k}: {exc}")


def pareto_mask(ndcg: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """No dominados maximizando (cov, ndcg)."""
    n = len(ndcg)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            if (cov[j] >= cov[i] and ndcg[j] >= ndcg[i]) and (
                cov[j] > cov[i] or ndcg[j] > ndcg[i]
            ):
                keep[i] = False
                break
    return keep


def plot_alpha_sweep(alpha, ndcg, cov, out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    l1 = ax1.plot(alpha, ndcg, "-o", color=C_NDCG, ms=4, lw=1.8, label="NDCG@10")
    ax1.set_xlabel(r"Peso de fusión $\alpha$")
    ax1.set_ylabel("NDCG@10 (relevancia)", color=C_NDCG)
    ax1.tick_params(axis="y", labelcolor=C_NDCG)
    ax1.set_xlim(0, 1)

    ax2 = ax1.twinx()
    ax2.spines.top.set_visible(False)
    l2 = ax2.plot(alpha, cov, "-s", color=C_COV, ms=4, lw=1.8, label="Cov@10")
    ax2.set_ylabel("Cov@10 (cobertura)", color=C_COV)
    ax2.tick_params(axis="y", labelcolor=C_COV)

    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_vs_beta(alpha, ndcg, cov, out: Path) -> None:
    betas = np.linspace(0, 1, 201)
    f = betas[:, None] * ndcg[None, :] + (1 - betas[:, None]) * cov[None, :]
    idx = f.argmax(axis=1)
    a_star = alpha[idx]
    ndcg_star = ndcg[idx]
    cov_star = cov[idx]

    fig, ax1 = plt.subplots(figsize=(6.6, 4.2))
    ax1.step(betas, a_star, where="mid", color=C_STAR, lw=2.2, label=r"$\alpha^{\star}(\beta)$")
    ax1.set_xlabel(r"Preferencia por relevancia $\beta$")
    ax1.set_ylabel(r"$\alpha^{\star}(\beta)$", color=C_STAR)
    ax1.tick_params(axis="y", labelcolor=C_STAR)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.03, 1.03)

    ax2 = ax1.twinx()
    ax2.spines.top.set_visible(False)
    ax2.plot(betas, ndcg_star, "--", color=C_NDCG, lw=1.8, label="NDCG@10")
    ax2.plot(betas, cov_star, ":", color=C_COV, lw=2.0, label="Cov@10")
    ax2.set_ylabel("Métrica en el óptimo", color="dimgray")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(alpha, ndcg, cov, out: Path) -> None:
    keep = pareto_mask(ndcg, cov)
    order = np.argsort(cov[keep])
    cov_f = cov[keep][order]
    ndcg_f = ndcg[keep][order]

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(cov[~keep], ndcg[~keep], c="lightgray", s=35, label="Dominados", zorder=2)
    ax.plot(cov_f, ndcg_f, "-", color=C_STAR, lw=1.6, zorder=3)
    ax.scatter(cov_f, ndcg_f, c=C_STAR, s=45, label="Frente de Pareto", zorder=4)

    # anotar algunos alfa representativos sobre el frente
    for a in (0.0, 0.3, 0.6, 0.8):
        i = int(np.argmin(np.abs(alpha - a)))
        if keep[i]:
            ax.annotate(
                rf"$\alpha={a:g}$",
                (cov[i], ndcg[i]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )
    ax.set_xlabel("Cov@10 (cobertura)")
    ax.set_ylabel("NDCG@10 (relevancia)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    alpha, ndcg, cov = load_sweep(args.k)

    keep = pareto_mask(ndcg, cov)
    print(f"K={args.k}: {len(alpha)} valores de alfa, {keep.sum()} no dominados")
    print("alfa no dominados:", ", ".join(f"{a:.2f}" for a in alpha[keep]))

    plot_alpha_sweep(alpha, ndcg, cov, FIG_DIR / "alpha_sweep.pdf")
    plot_alpha_vs_beta(alpha, ndcg, cov, FIG_DIR / "alphaVSbeta.pdf")
    plot_pareto(alpha, ndcg, cov, FIG_DIR / "pareto_realized.pdf")
    print(f"figuras escritas en {FIG_DIR}/")


if __name__ == "__main__":
    main()
