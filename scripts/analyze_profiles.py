"""
Análisis de los perfiles de aspecto de los usuarios (matriz X, variante D).

Parte 1 — Clustering (dos representaciones):
  a) X cruda estandarizada por aspecto: captura la estructura dominante
     (intensidad de mención).
  b) X normalizada L1 por fila (los pesos w_u que usa CBQuality),
     estandarizada: captura la orientación temática relativa.
  La selección de k usa el coeficiente de silueta; los centroides se
  reportan en la escala interpretable de cada representación. Se cruza la
  composición de tipos de establecimiento reseñados por cluster.

Parte 2 — Coherencia perfil-visitas:
  coh(u, S) = corr_Pearson(X_u, media de Y sobre los pueblos de S), para
  S = historial de entrenamiento, pueblos de prueba, top-10 de CBQuality y
  top-10 de CFAttention. Cada conjunto se contrasta con un baseline nulo de
  conjuntos aleatorios DEL MISMO TAMAÑO (la media de Y sobre conjuntos
  grandes se acerca al perfil medio global y sube la correlación de forma
  espuria, así que el tamaño debe emparejarse). Se reporta el delta
  coh(S) − coh(aleatorio emparejado) por usuario.

Requiere haber corrido antes: scripts/analyze_hybrid_sweep.py
(genera los top-10 por modelo en reports/results/analisis/).

Uso:
    uv run python scripts/analyze_profiles.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from analysis_common import (
    FIG_DIR,
    MODEL_COLORS,
    RANDOM_STATE,
    RESULTS_DIR,
    build_ground_truth,
    load_matrices,
    load_splits,
    setup_style,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

K_RANGE = range(2, 11)
N_RANDOM_DRAWS = 30

# Okabe-Ito: paleta categórica segura para daltonismo (hasta 8 clusters).
CLUSTER_COLORS = [
    "#0173B2", "#DE8F05", "#029E73", "#CC78BC",
    "#56B4E9", "#D55E00", "#F0E442", "#949494",
]


# --------------------------------------------------------------------------- #
# Parte 1 — Clustering                                                         #
# --------------------------------------------------------------------------- #


def _kmeans_with_selection(Z: np.ndarray, tag: str) -> tuple[pd.Series, list[float]]:
    silhouettes = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        silhouettes.append(silhouette_score(Z, km.labels_))
        logger.info(f"  [{tag}] k={k}: silueta={silhouettes[-1]:.4f}")
    k_best = list(K_RANGE)[int(np.argmax(silhouettes))]
    logger.info(f"  [{tag}] k óptimo por silueta: {k_best}")
    km = KMeans(n_clusters=k_best, n_init=10, random_state=RANDOM_STATE).fit(Z)
    # Reordenar etiquetas por tamaño descendente para lectura estable
    order = pd.Series(km.labels_).value_counts().index.tolist()
    relabel = {old: new for new, old in enumerate(order)}
    labels = pd.Series([relabel[c] for c in km.labels_], name="cluster")
    return labels, silhouettes


def _centroid_heatmap(
    centroids: pd.DataFrame,
    sizes: pd.Series,
    vmin: float,
    vmax: float,
    cbar_label: str,
    out: Path,
    fmt: str = "{:.2f}",
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(centroids) + 1.6))
    im = ax.imshow(centroids.values, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(centroids.columns)))
    ax.set_xticklabels(centroids.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(centroids)))
    ax.set_yticklabels([f"C{c} (n={sizes[c]:,})" for c in centroids.index])
    thresh = vmin + 0.6 * (vmax - vmin)
    for i in range(centroids.shape[0]):
        for j in range(centroids.shape[1]):
            v = centroids.values[i, j]
            ax.text(
                j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                color="white" if v > thresh else "#1a1a2e",
            )
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _tipo_composition(train: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    tipo_frac = train.groupby(["Author", "Tipo"]).size().unstack(fill_value=0)
    tipo_frac = tipo_frac.div(tipo_frac.sum(axis=1), axis=0)
    return tipo_frac.groupby(labels.reindex(tipo_frac.index)).mean()


def run_clustering(X: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    # ── a) X cruda: estructura de intensidad ─────────────────────────────── #
    Z_raw = StandardScaler().fit_transform(X.values)
    labels_raw, sil_raw = _kmeans_with_selection(Z_raw, "cruda")
    labels_raw.index = X.index

    # ── b) X normalizada L1 (pesos de CBQuality): orientación temática ───── #
    W = X.div(X.sum(axis=1), axis=0)
    Z_norm = StandardScaler().fit_transform(W.values)
    labels_norm, sil_norm = _kmeans_with_selection(Z_norm, "normalizada")
    labels_norm.index = X.index

    # ── Figura: selección de k (silueta para ambas representaciones) ─────── #
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(list(K_RANGE), sil_raw, "-o", color="#0173B2", ms=4,
            label="$X$ cruda (intensidad)")
    ax.plot(list(K_RANGE), sil_norm, "-s", color="#CC78BC", ms=4,
            label="$X$ normalizada (orientación)")
    ax.set_xlabel("Número de clusters $k$")
    ax.set_ylabel("Coeficiente de silueta")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "profiles_cluster_selection.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Figuras: centroides ──────────────────────────────────────────────── #
    cent_raw = X.groupby(labels_raw).mean()
    sizes_raw = labels_raw.value_counts().sort_index()
    _centroid_heatmap(
        cent_raw, sizes_raw, 1, 5, "Importancia media $X_{u,a}$",
        FIG_DIR / "profiles_centroids.pdf",
    )

    cent_norm = W.groupby(labels_norm).mean()
    sizes_norm = labels_norm.value_counts().sort_index()
    _centroid_heatmap(
        cent_norm, sizes_norm, cent_norm.values.min(), cent_norm.values.max(),
        "Peso relativo medio $w_{u,a}$",
        FIG_DIR / "profiles_centroids_norm.pdf", fmt="{:.3f}",
    )

    # ── Figura: proyección PCA de la representación temática ────────────── #
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    P = pca.fit_transform(Z_norm)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for c in sorted(labels_norm.unique()):
        mask = labels_norm.values == c
        ax.scatter(
            P[mask, 0], P[mask, 1], s=6, alpha=0.35, lw=0,
            color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
            label=f"C{c} (n={mask.sum():,})",
        )
    ax.set_xlabel(f"CP1 ({pca.explained_variance_ratio_[0]:.0%} de la varianza)")
    ax.set_ylabel(f"CP2 ({pca.explained_variance_ratio_[1]:.0%} de la varianza)")
    leg = ax.legend(frameon=False, markerscale=2.5, fontsize=9)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "profiles_pca.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Resúmenes ────────────────────────────────────────────────────────── #
    summary_raw = pd.concat(
        [sizes_raw.rename("n_usuarios"), cent_raw, _tipo_composition(train, labels_raw)],
        axis=1,
    )
    summary_raw.to_csv(RESULTS_DIR / "profiles_cluster_summary.csv")

    summary_norm = pd.concat(
        [sizes_norm.rename("n_usuarios"), cent_norm,
         _tipo_composition(train, labels_norm)],
        axis=1,
    )
    summary_norm.to_csv(RESULTS_DIR / "profiles_cluster_summary_norm.csv")

    assignments = pd.DataFrame(
        {"cluster_intensidad": labels_raw, "cluster_tema": labels_norm}
    )
    assignments.index.name = "user"
    assignments.to_csv(RESULTS_DIR / "profiles_cluster_assignments.csv")

    print("\nClusters de intensidad (X cruda, centroides en [1,5] + tipos):")
    print(summary_raw.round(3).to_string())
    print("\nClusters temáticos (X normalizada, pesos relativos + tipos):")
    print(summary_norm.round(3).to_string())
    print(f"\nVarianza explicada PCA 2D (temática): {pca.explained_variance_ratio_.sum():.1%}")
    return labels_norm


# --------------------------------------------------------------------------- #
# Parte 2 — Coherencia perfil-visitas                                          #
# --------------------------------------------------------------------------- #


def _pearson_rows(xu: np.ndarray, ybar: np.ndarray) -> float:
    if np.std(xu) == 0 or np.std(ybar) == 0:
        return np.nan
    return float(np.corrcoef(xu, ybar)[0, 1])


def run_coherence(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    A: pd.DataFrame,
    gt: dict[str, set[str]],
) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    aspects = X.columns.intersection(Y.columns)
    Xv = X[aspects]
    Yv = Y[aspects]
    pueblos = np.array(Yv.index)
    y_lookup = {p: Yv.loc[p].values for p in Yv.index}

    def coh(user: str, subset: list[str]) -> float:
        vecs = [y_lookup[p] for p in subset if p in y_lookup]
        if not vecs:
            return np.nan
        return _pearson_rows(Xv.loc[user].values, np.mean(vecs, axis=0))

    topk_cb = (
        pd.read_csv(RESULTS_DIR / "topk10_CBQuality.csv")
        .groupby("user")["pueblo"].apply(list).to_dict()
    )
    topk_cf = (
        pd.read_csv(RESULTS_DIR / "topk10_CFAttention.csv")
        .groupby("user")["pueblo"].apply(list).to_dict()
    )
    visited_map = {u: list(row.dropna().index) for u, row in A.iterrows()}

    def null_mean(user: str, size: int) -> float:
        vals = [
            coh(user, list(rng.choice(pueblos, size=size, replace=False)))
            for _ in range(N_RANDOM_DRAWS)
        ]
        return float(np.nanmean(vals))

    rows = {}
    for user in gt:
        hist = visited_map.get(user, [])
        future = list(gt[user])
        sets = {
            "historial": hist,
            "prueba": future,
            "top10_cb": topk_cb[user],
            "top10_cf": topk_cf[user],
        }
        row: dict[str, float] = {"n_historial": len(hist)}
        for name, subset in sets.items():
            c = coh(user, subset)
            base = null_mean(user, max(len(subset), 1))
            row[name] = c
            row[f"{name}_nulo"] = base
            row[f"{name}_delta"] = c - base
        rows[user] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "user"
    df.to_csv(RESULTS_DIR / "profiles_coherence.csv")

    # ── Pruebas pareadas contra el baseline emparejado por tamaño ────────── #
    print("\nCoherencia perfil-visitas (baseline aleatorio del mismo tamaño):")
    tests = {}
    for col in ["historial", "prueba", "top10_cb", "top10_cf"]:
        pair = df[[col, f"{col}_nulo"]].dropna()
        res = stats.wilcoxon(
            pair[col], pair[f"{col}_nulo"], zero_method="pratt", method="approx"
        )
        n = len(pair)
        r_eff = abs(res.zstatistic) / np.sqrt(n)
        tests[col] = {
            "mediana": pair[col].median(),
            "mediana_nulo": pair[f"{col}_nulo"].median(),
            "mediana_delta": df[f"{col}_delta"].median(),
            "wilcoxon_p": res.pvalue,
            "efecto_r": r_eff,
            "n": n,
        }
        print(
            f"  {col:<10} mediana={pair[col].median():.3f}  "
            f"nulo={pair[f'{col}_nulo'].median():.3f}  "
            f"delta={df[f'{col}_delta'].median():+.3f}  "
            f"p={res.pvalue:.2e}  r={r_eff:.3f}"
        )
    pd.DataFrame(tests).T.to_csv(RESULTS_DIR / "profiles_coherence_tests.csv")

    # ── Figura: distribuciones del delta de coherencia ───────────────────── #
    order = [
        ("historial_delta", "Historial", "#0173B2"),
        ("prueba_delta", "Visitas futuras", "#CC78BC"),
        ("top10_cb_delta", "Top-10 CBQuality", MODEL_COLORS["CBQuality"]),
        ("top10_cf_delta", "Top-10 CFAttention", MODEL_COLORS["CFAttention"]),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    data = [df[c].dropna().values for c, _, _ in order]
    parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.82)
    for body, (_, _, color) in zip(parts["bodies"], order):
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor("none")
    parts["cmedians"].set_color("#1a1a2e")
    parts["cmedians"].set_linewidth(1.4)
    ax.axhline(0, color="#949494", lw=1.0, ls="--")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([lbl for _, lbl, _ in order], fontsize=9.5)
    ax.set_ylabel(r"$\Delta$ coherencia vs. aleatorio emparejado")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "profiles_coherence.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    A, X, Y = load_matrices()
    train, test = load_splits()
    gt = build_ground_truth(A, test)

    logger.info("Parte 1 — clustering de perfiles…")
    run_clustering(X, train)

    logger.info("Parte 2 — coherencia perfil-visitas…")
    run_coherence(X, Y, A, gt)

    logger.info("Listo.")


if __name__ == "__main__":
    main()
