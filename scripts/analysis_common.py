"""
Utilidades compartidas por los scripts de análisis de resultados (analyze_*.py).

Centraliza: rutas canónicas (matrices zs-bert-tempmedian, splits, figuras),
carga del híbrido serializado (con shim para el rename CBAttention→CFAttention),
extracción de matrices de scores por rama, top-K vectorizado con exclusión de
visitados y métricas por usuario calculadas desde listas.

Los colores de modelo son fijos en todas las figuras del capítulo de resultados
(paleta validada para daltonismo; el estilo base replica plot_hybrid_figures.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from mtrs.eval.metrics import (  # noqa: E402
    hit_at_k,
    intra_list_diversity,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

DATA_DIR = Path("data/rest-mex")
# Variante D de la ablación (la adoptada en la tesis): X con escalamiento T,
# Y sin compresión logarítmica. Ver scripts/build_matrices.py.
MATRICES_DIR = DATA_DIR / "matrices" / "zs-bert-d"
SPLITS_DIR = DATA_DIR / "splits"
FIG_DIR = Path("reports/Tesis/figures")
RESULTS_DIR = Path("reports/results/analisis")
HYBRID_PATH = Path("models/hybrid_ab/hybrid.joblib")

# Colores fijos por modelo (validados: CVD-safe sobre fondo claro).
MODEL_COLORS = {
    "CBQuality": "#0173B2",
    "CFAttention": "#DE8F05",
    "CFClassic": "#029E73",
    "Híbrido": "#C44E52",
}

RANDOM_STATE = 42


def setup_style() -> None:
    """Estilo base de las figuras del capítulo (idéntico a plot_hybrid_figures)."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )


# --------------------------------------------------------------------------- #
# Carga de datos y modelos                                                     #
# --------------------------------------------------------------------------- #


def _install_cb_attention_shim() -> None:
    """Alias de módulo para deserializar joblibs previos al rename CBAttention."""
    import mtrs.models.cf_attention as cfa

    sys.modules.setdefault("mtrs.models.cb_attention", cfa)
    if not hasattr(cfa, "CBAttention"):
        cfa.CBAttention = cfa.CFAttention


def load_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    A = pd.read_parquet(MATRICES_DIR / "A.parquet")
    X = pd.read_parquet(MATRICES_DIR / "X.parquet")
    Y = pd.read_parquet(MATRICES_DIR / "Y.parquet")
    return A, X, Y


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    test = pd.read_parquet(SPLITS_DIR / "test.parquet")
    return train, test


def build_ground_truth(A: pd.DataFrame, test: pd.DataFrame) -> dict[str, set[str]]:
    """Pueblos de prueba por usuario, restringido a usuarios presentes en A."""
    test = test[test["Author"].isin(A.index)]
    return test.groupby("Author")["Pueblo"].apply(set).to_dict()


def load_hybrid():
    _install_cb_attention_shim()
    return joblib.load(HYBRID_PATH)


def branch_scores(hybrid) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matrices de scores (usuarios × pueblos) de cada rama del híbrido.

    El stack serializado conserva el orden [CFAttention, CBQuality]
    (nombres previos al rename: ['CBAttention', 'CBQuality']).
    """
    users, pueblos = hybrid._users, hybrid._pueblos
    s_cf = pd.DataFrame(hybrid._score_stack[0], index=users, columns=pueblos)
    s_cb = pd.DataFrame(hybrid._score_stack[1], index=users, columns=pueblos)
    return s_cf, s_cb


def fused_scores(s_cf: pd.DataFrame, s_cb: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """ŷ(α) = α·ŷ_CFAttention + (1−α)·ŷ_CBQuality, recortado a [1, 5]."""
    vals = np.clip(alpha * s_cf.values + (1.0 - alpha) * s_cb.values, 1.0, 5.0)
    return pd.DataFrame(vals, index=s_cf.index, columns=s_cf.columns)


# --------------------------------------------------------------------------- #
# Scoring vectorizado (para refits con X modificada, p. ej. arranque en frío)  #
# --------------------------------------------------------------------------- #


def cb_quality_scores(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """r̂ = (X_u/‖X_u‖₁)·Y_p — réplica vectorizada de CBQuality.fit()."""
    aspects = X.columns.intersection(Y.columns)
    Xa, Ya = X[aspects], Y[aspects]
    W = Xa.div(Xa.sum(axis=1).replace(0, 1), axis=0)
    return pd.DataFrame(W.values @ Ya.values.T, index=X.index, columns=Y.index)


def cf_attention_scores(
    X: pd.DataFrame,
    A: pd.DataFrame,
    k_neighbors: int = 20,
) -> pd.DataFrame:
    """Réplica vectorizada del scoring de CFAttention (min_sim=0).

    Para cada pueblo p toma, entre los usuarios que lo observaron, los k vecinos
    más similares a cada usuario y predice μ_u + Σ w·(A_vp − μ_v) / Σ w.
    """
    from scipy.spatial.distance import cdist

    users, pueblos = list(A.index), list(A.columns)
    X_aligned = X.reindex(index=users).fillna(1.0)
    sim = (1.0 / (1.0 + cdist(X_aligned.values, X_aligned.values))).astype(np.float32)
    np.fill_diagonal(sim, 0.0)

    A_vals = A.reindex(index=users, columns=pueblos).values.astype(float)
    mu = np.nanmean(A_vals, axis=1)
    mu_filled = np.where(np.isnan(mu), 3.0, mu)

    scores = np.tile(mu_filled[:, None], (1, len(pueblos)))
    for pi in range(len(pueblos)):
        col = A_vals[:, pi]
        observed = np.where(~np.isnan(col))[0]
        if observed.size == 0:
            continue
        sub = sim[:, observed]  # (m, |O_p|)
        k = min(k_neighbors, observed.size)
        if k < observed.size:
            idx = np.argpartition(-sub, k, axis=1)[:, :k]
        else:
            idx = np.tile(np.arange(observed.size), (sub.shape[0], 1))
        w = np.take_along_axis(sub, idx, axis=1)  # (m, k)
        dev = (col - mu_filled)[observed][idx]  # (m, k)
        valid = w > 0
        w = np.where(valid, w, 0.0)
        dev = np.where(valid, dev, 0.0)
        wsum = w.sum(axis=1)
        has = wsum > 0
        scores[has, pi] = mu_filled[has] + (w * dev).sum(axis=1)[has] / wsum[has]
    return pd.DataFrame(scores, index=users, columns=pueblos)


# --------------------------------------------------------------------------- #
# Top-K y métricas desde matrices de scores                                    #
# --------------------------------------------------------------------------- #


def topk_lists(
    scores: pd.DataFrame,
    A: pd.DataFrame,
    k: int,
    users: list[str] | None = None,
) -> dict[str, list[str]]:
    """Top-K por usuario excluyendo visitados (celdas observadas de A).

    Replica BaseRecommender.recommend(): orden descendente por score con
    desempate estable por orden de columna.
    """
    if users is None:
        users = list(scores.index)
    vals = scores.loc[users].values.astype(float).copy()
    visited = A.reindex(index=users, columns=scores.columns).notna().values
    vals[visited] = -np.inf
    order = np.argsort(-vals, axis=1, kind="stable")[:, :k]
    cols = np.asarray(scores.columns)
    return {u: [str(p) for p in cols[order[i]]] for i, u in enumerate(users)}


def per_user_metrics(
    topk: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    Y: pd.DataFrame,
) -> pd.DataFrame:
    """Métricas top-K por usuario a partir de listas ya generadas."""
    rows = {}
    for user, relevant in ground_truth.items():
        recs = topk[user]
        rows[user] = {
            "hit": hit_at_k(recs, relevant),
            "precision": precision_at_k(recs, relevant),
            "recall": recall_at_k(recs, relevant),
            "ndcg": ndcg_at_k(recs, relevant),
            "mrr": mrr(recs, relevant),
            "ild": intra_list_diversity(recs, Y),
            "n_relevant": float(len(relevant)),
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "user"
    return df


def catalog_coverage(topk: dict[str, list[str]], n_pueblos: int) -> float:
    recommended: set[str] = set()
    for recs in topk.values():
        recommended.update(recs)
    return len(recommended) / n_pueblos if n_pueblos else 0.0
