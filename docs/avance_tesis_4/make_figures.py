"""Genera las figuras para la presentacion Avance de Tesis 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

CREAM = "#F2EEE3"
NAVY = "#2F3E4E"

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
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    }
)

# ───────────────────────────── Datos ablacion ───────────────────────────── #
# Columnas: variant, model, K, hit, precision, recall, ndcg, mrr, ild, coverage

raw = [
    # A — CBAttention
    ("A", "CBAttention", 1,  0.0300, 0.0300, 0.0267, 0.0300, 0.0300, 0.0000, 0.9792),
    ("A", "CBAttention", 3,  0.0894, 0.0301, 0.0810, 0.0591, 0.0550, 0.0027, 0.9792),
    ("A", "CBAttention", 5,  0.1445, 0.0293, 0.1321, 0.0804, 0.0674, 0.0027, 0.9792),
    ("A", "CBAttention", 10, 0.2821, 0.0289, 0.2611, 0.1224, 0.0851, 0.0027, 1.0000),
    ("A", "CBAttention", 20, 0.5521, 0.0292, 0.5256, 0.1902, 0.1033, 0.0026, 1.0000),
    # A — CBQuality
    ("A", "CBQuality", 1,  0.1574, 0.1574, 0.1436, 0.1574, 0.1574, 0.0000, 0.2500),
    ("A", "CBQuality", 3,  0.3349, 0.1162, 0.3083, 0.2435, 0.2316, 0.0009, 0.3542),
    ("A", "CBQuality", 5,  0.4442, 0.0962, 0.4245, 0.2926, 0.2565, 0.0010, 0.4583),
    ("A", "CBQuality", 10, 0.5714, 0.0626, 0.5544, 0.3358, 0.2736, 0.0015, 0.6458),
    ("A", "CBQuality", 20, 0.7781, 0.0425, 0.7615, 0.3883, 0.2874, 0.0018, 0.9583),
    # B — CBAttention
    ("B", "CBAttention", 1,  0.0301, 0.0301, 0.0264, 0.0301, 0.0301, 0.0000, 0.9583),
    ("B", "CBAttention", 3,  0.0898, 0.0301, 0.0804, 0.0586, 0.0549, 0.0043, 0.9792),
    ("B", "CBAttention", 5,  0.1437, 0.0290, 0.1303, 0.0793, 0.0670, 0.0046, 1.0000),
    ("B", "CBAttention", 10, 0.2871, 0.0294, 0.2650, 0.1232, 0.0855, 0.0050, 1.0000),
    ("B", "CBAttention", 20, 0.5484, 0.0289, 0.5208, 0.1888, 0.1032, 0.0056, 1.0000),
    # B — CBQuality
    ("B", "CBQuality", 1,  0.0877, 0.0877, 0.0807, 0.0877, 0.0877, 0.0000, 0.2083),
    ("B", "CBQuality", 3,  0.2067, 0.0697, 0.1949, 0.1443, 0.1326, 0.0000, 0.3750),
    ("B", "CBQuality", 5,  0.3116, 0.0636, 0.2862, 0.1827, 0.1556, 0.0000, 0.4583),
    ("B", "CBQuality", 10, 0.5988, 0.0650, 0.5794, 0.2778, 0.1918, 0.0000, 0.6458),
    ("B", "CBQuality", 20, 0.7752, 0.0421, 0.7566, 0.3238, 0.2043, 0.0000, 0.8958),
    # C — CBAttention
    ("C", "CBAttention", 1,  0.0301, 0.0301, 0.0264, 0.0301, 0.0301, 0.0000, 0.9583),
    ("C", "CBAttention", 3,  0.0898, 0.0301, 0.0804, 0.0586, 0.0549, 0.0073, 0.9792),
    ("C", "CBAttention", 5,  0.1437, 0.0290, 0.1303, 0.0793, 0.0670, 0.0073, 1.0000),
    ("C", "CBAttention", 10, 0.2871, 0.0294, 0.2650, 0.1232, 0.0855, 0.0072, 1.0000),
    ("C", "CBAttention", 20, 0.5484, 0.0289, 0.5208, 0.1888, 0.1032, 0.0071, 1.0000),
    # C — CBQuality
    ("C", "CBQuality", 1,  0.1805, 0.1805, 0.1658, 0.1805, 0.1805, 0.0000, 0.2500),
    ("C", "CBQuality", 3,  0.3429, 0.1193, 0.3172, 0.2575, 0.2471, 0.0003, 0.3958),
    ("C", "CBQuality", 5,  0.4642, 0.1016, 0.4469, 0.3127, 0.2752, 0.0010, 0.4792),
    ("C", "CBQuality", 10, 0.6285, 0.0686, 0.6116, 0.3666, 0.2967, 0.0038, 0.7083),
    ("C", "CBQuality", 20, 0.8254, 0.0452, 0.8113, 0.4180, 0.3104, 0.0077, 0.9583),
    # D — CBAttention
    ("D", "CBAttention", 1,  0.0300, 0.0300, 0.0267, 0.0300, 0.0300, 0.0000, 0.9792),
    ("D", "CBAttention", 3,  0.0894, 0.0301, 0.0810, 0.0591, 0.0550, 0.0076, 0.9792),
    ("D", "CBAttention", 5,  0.1445, 0.0293, 0.1321, 0.0804, 0.0674, 0.0075, 0.9792),
    ("D", "CBAttention", 10, 0.2821, 0.0289, 0.2611, 0.1224, 0.0851, 0.0074, 1.0000),
    ("D", "CBAttention", 20, 0.5521, 0.0292, 0.5256, 0.1902, 0.1033, 0.0072, 1.0000),
    # D — CBQuality
    ("D", "CBQuality", 1,  0.1810, 0.1810, 0.1662, 0.1810, 0.1810, 0.0000, 0.2500),
    ("D", "CBQuality", 3,  0.3433, 0.1197, 0.3178, 0.2578, 0.2472, 0.0003, 0.3958),
    ("D", "CBQuality", 5,  0.4628, 0.1016, 0.4462, 0.3124, 0.2747, 0.0010, 0.4792),
    ("D", "CBQuality", 10, 0.6269, 0.0684, 0.6100, 0.3658, 0.2962, 0.0037, 0.7292),
    ("D", "CBQuality", 20, 0.8254, 0.0452, 0.8110, 0.4178, 0.3101, 0.0077, 0.9583),
]

df = pd.DataFrame(
    raw,
    columns=["variant", "model", "K", "hit", "precision", "recall", "ndcg", "mrr", "ild", "coverage"],
)

# Diagnosticos de matrices
diag = pd.DataFrame(
    [
        ("A", 2.308, 1.001, 0.096, 3.895, 0.094, 0.013),
        ("B", 3.087, 1.884, 0.365, 4.937, 0.154, 0.979),
        ("C", 3.087, 1.884, 0.365, 4.043, 0.225, 0.336),
        ("D", 2.308, 1.001, 0.096, 4.043, 0.225, 0.336),
    ],
    columns=["variant", "X_mean", "X_row_var", "X_sat_hi", "Y_mean", "Y_row_var", "Y_sat_hi"],
)

variant_colors = {"A": "#7E57C2", "B": "#EF6C00", "C": "#2E7D32", "D": "#C62828"}
variant_marker = {"A": "o", "B": "s", "C": "^", "D": "D"}

variant_label = {
    "A": "A: sigm+T / log+T (actual)",
    "B": "B: Zhang puros",
    "C": "C: Zhang X / Zhang Y + T",
    "D": "D: sigm+T X / Zhang Y + T",
}

# ───────────── Fig 1: NDCG@K por variante para CBQuality ─────────────────── #
def plot_metric_by_variant(metric: str, model: str, ylabel: str, filename: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sub = df[df["model"] == model]
    for variant in ["A", "B", "C", "D"]:
        s = sub[sub["variant"] == variant].sort_values("K")
        ax.plot(
            s["K"], s[metric],
            color=variant_colors[variant],
            marker=variant_marker[variant],
            label=variant_label[variant],
            linewidth=2, markersize=7,
        )
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} - {model}")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks([1, 3, 5, 10, 20])
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=160)
    plt.close(fig)


plot_metric_by_variant("ndcg", "CBQuality", "NDCG@K", "ndcg_cbquality.png")
plot_metric_by_variant("coverage", "CBQuality", "Coverage@K", "cov_cbquality.png")
plot_metric_by_variant("ndcg", "CBAttention", "NDCG@K", "ndcg_cbattention.png")


# ───────────── Fig 2: Diagnostico de Y (saturacion alta) ─────────────────── #
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
xs = np.arange(len(diag))
axes[0].bar(xs, diag["Y_sat_hi"], color=[variant_colors[v] for v in diag["variant"]])
axes[0].set_xticks(xs)
axes[0].set_xticklabels(diag["variant"])
axes[0].set_ylabel("Frac. celdas $Y_{p,j}>4.5$")
axes[0].set_title("Saturacion alta de Y")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
axes[0].set_ylim(0, 1.0)

axes[1].bar(xs, diag["Y_row_var"], color=[variant_colors[v] for v in diag["variant"]])
axes[1].set_xticks(xs)
axes[1].set_xticklabels(diag["variant"])
axes[1].set_ylabel("Varianza promedio por fila de Y")
axes[1].set_title("Discriminacion entre pueblos")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "y_diagnostics.png", dpi=160)
plt.close(fig)


# ───────────── Fig 3: Resumen comparativo K=10 ─────────────────────────────#
metrics_to_plot = [("ndcg", "NDCG@10"), ("hit", "Hit@10"), ("mrr", "MRR@10"), ("coverage", "Cov@10")]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, (m, title) in zip(axes, metrics_to_plot):
    sub = df[(df["model"] == "CBQuality") & (df["K"] == 10)].sort_values("variant")
    ax.bar(sub["variant"], sub[m], color=[variant_colors[v] for v in sub["variant"]])
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(sub[m].values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(sub[m]) * 1.15)
fig.suptitle("CBQuality @ K=10 - comparacion entre variantes", y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "summary_k10.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Figuras generadas en {FIG_DIR}")
