# Construcción de las matrices estructurales del sistema de recomendación.
#
# Tres matrices alimentan los modelos basados en aspectos: A (ratings),
# X (importancia usuario-aspecto) e Y (calidad pueblo-aspecto).
#
# Notación:
#   u  — usuario (Author)
#   p  — pueblo mágico (Pueblo)
#   j  — aspecto (servicio, comida, ...) — 8 únicos en ALL_ASPECTS
#   N  — escala de calificación (1–5)
#
# Entrada esperada: DataFrame de salida de AspectSentimentExtractor con columnas:
#   Author, Pueblo, Lugar, Tipo, Calificacion, review_idx,
#   sentence, aspect, aspect_score, sentiment, sentiment_score

import numpy as np
import pandas as pd

from mtrs.aspects.config import ALL_ASPECTS, LABEL2ID_SENTIMENT

N: int = 5  # Escala de calificación


def _signed_sentiment(df: pd.DataFrame) -> pd.Series:
    """Mapea etiqueta de sentimiento a puntuación en [-2, 2].

    sent_score(s) = LABEL2ID[sentiment(s)] - 2
    """
    return df["sentiment"].map(LABEL2ID_SENTIMENT) - 2


# ---------------------------------------------------------------------------
# A_{u,p}  —  Matriz de Ratings
# ---------------------------------------------------------------------------


def compute_rating_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la matriz de ratings promedio por usuario y pueblo.

    A_{u,p} = (1 / |R_{u,p}|) * sum_{r in R_{u,p}} Calificacion(r)

    Los chunks se deduplican por review_idx para evitar contar varias veces
    la misma reseña. El resultado es un DataFrame (Author x Pueblo) con NaN
    donde el usuario no visitó el pueblo.

    Returns:
        pd.DataFrame: Shape (n_users, n_pueblos), valores en [1, 5].
    """
    unique_reviews = df.drop_duplicates(subset="review_idx")[
        ["Author", "Pueblo", "Calificacion"]
    ]
    A = (
        unique_reviews.groupby(["Author", "Pueblo"])["Calificacion"]
        .mean()
        .unstack(level="Pueblo")
    )
    return A


# ---------------------------------------------------------------------------
# X_{u,j}  —  Importancia usuario–aspecto
# ---------------------------------------------------------------------------


def compute_user_aspect_importance(
    df: pd.DataFrame, T: float | None = None
) -> pd.DataFrame:
    """Calcula la importancia que cada usuario asigna a cada aspecto.

    t_{u,j} = número de chunks del usuario u clasificados en aspecto j
              (sobre todos los pueblos).

    X_{u,j} = 1 + (N-1) * (2 / (1 + exp(-t_{u,j} / T)) - 1)

    La función sigmoide desplazada mapea t in [0, inf) -> X in [1, N]:
      - t=0   => X=1   (aspecto nunca mencionado)
      - t->inf => X->N (aspecto muy frecuente)

    La temperatura T ancla el punto de inflexión (X = (N+1)/2 = 3) a la
    mediana de los conteos no nulos: un usuario con el número típico de
    menciones obtiene importancia media.

    Args:
        df: DataFrame con columnas ['Author', 'aspect'].
            Cada fila representa un chunk clasificado en un aspecto por un usuario.
        T: Temperatura para ajustar la sensibilidad de la función sigmoide.
           Si es None (default), se deriva como median(t_{u,j} | t_{u,j} > 0).

    Returns:
        pd.DataFrame: Shape (n_users, len(ALL_ASPECTS)), valores en [1, 5].
    """
    t = (
        df.groupby(["Author", "aspect"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ALL_ASPECTS, fill_value=0)
    )
    if T is None:
        nonzero = t.values[t.values > 0]
        T = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0
    sigmoid_shifted = 2.0 / (1.0 + np.exp(-t.values / T)) - 1.0
    X = pd.DataFrame(
        1.0 + (N - 1) * sigmoid_shifted,
        index=t.index,
        columns=t.columns,
    )
    return X


# ---------------------------------------------------------------------------
# Y_{p,j}  —  Calidad pueblo–aspecto
# ---------------------------------------------------------------------------


def compute_pueblo_aspect_quality(
    df: pd.DataFrame, T: float | None = None
) -> pd.DataFrame:
    """Calcula la calidad de cada pueblo en cada aspecto.

    h_{p,j}   = sentimiento promedio (en [-2, 2]) para pueblo p, aspecto j.
    phi_{p,j} = log(1 + |S_{p,j}|)  — volumen logarítmico de menciones.

    Y_{p,j} = 1 + (N-1) / (1 + exp(-h_{p,j} * phi_{p,j} / T))

    El producto h * phi actúa como señal volumen-ponderada:
      - Muchas menciones positivas -> Y -> N (alta calidad)
      - Pocas menciones o neutras  -> Y ~ (N+1)/2 (calidad media)
      - Menciones negativas        -> Y -> 1 (baja calidad)

    La temperatura T ancla el punto de inflexión de Y al valor absoluto
    mediano de la señal h*phi no nula: la señal típica produce el máximo
    gradiente de discriminación.

    Args:
        df: DataFrame con columnas ['Pueblo', 'aspect', 'sentiment'].
             Cada fila representa un chunk clasificado en un aspecto con un sentimiento.
        T: Temperatura para ajustar la sensibilidad de la función sigmoide.
           Si es None (default), se deriva como median(|h_{p,j} * phi_{p,j}| != 0).

    Returns:
        pd.DataFrame: Shape (n_pueblos, len(ALL_ASPECTS)), valores en [1, 5].
    """
    signed = df.copy()
    signed["sent_score"] = _signed_sentiment(signed)

    grouped = signed.groupby(["Pueblo", "aspect"])

    h = (
        grouped["sent_score"]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(columns=ALL_ASPECTS, fill_value=0.0)
    )
    counts = (
        grouped.size().unstack(fill_value=0).reindex(columns=ALL_ASPECTS, fill_value=0)
    )
    phi = np.log1p(counts.values)

    signal = h.values * phi
    if T is None:
        nonzero_signal = np.abs(signal[signal != 0])
        T = float(np.median(nonzero_signal)) if len(nonzero_signal) > 0 else 1.0

    exponent = signal / T
    Y = pd.DataFrame(
        1.0 + (N - 1) / (1.0 + np.exp(-exponent)),
        index=h.index,
        columns=h.columns,
    )
    return Y


# ---------------------------------------------------------------------------
# Variantes de ablación — fórmulas literales / intermedias de Zhang (2014)
# ---------------------------------------------------------------------------


def compute_pueblo_aspect_quality_zhang(df: pd.DataFrame) -> pd.DataFrame:
    """Y_{p,j} fiel a Zhang et al. (2014), Eq. 3 — sin log-volumen ni temperatura.

    Y_{p,j} = 1 + (N-1) / (1 + exp(-t_{p,j} * s_{p,j}))

    donde:
      t_{p,j} = conteo crudo de chunks de pueblo p sobre aspecto j.
      s_{p,j} = sentimiento promedio escalado a [-1, 1].

    El sentimiento se reescala dividiendo por 2 para ajustarse al rango
    asumido en el paper original (en `_signed_sentiment` está en [-2, 2]).

    Referencia:
        Zhang et al. (2014), "Explicit Factor Models for Explainable
        Recommendation based on Phrase-level Sentiment Analysis",
        SIGIR '14, Eq. 3.

    Returns:
        pd.DataFrame: Shape (n_pueblos, len(ALL_ASPECTS)), valores en [1, 5].
    """
    signed = df.copy()
    signed["sent_score"] = _signed_sentiment(signed) / 2.0  # → [-1, 1]

    grouped = signed.groupby(["Pueblo", "aspect"])
    s = (
        grouped["sent_score"]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(columns=ALL_ASPECTS, fill_value=0.0)
    )
    t = (
        grouped.size().unstack(fill_value=0).reindex(columns=ALL_ASPECTS, fill_value=0)
    )

    exponent = t.values * s.values
    Y = pd.DataFrame(
        1.0 + (N - 1) / (1.0 + np.exp(-exponent)),
        index=s.index,
        columns=s.columns,
    )
    return Y


def compute_pueblo_aspect_quality_zhang_T(
    df: pd.DataFrame, T: float | None = None
) -> pd.DataFrame:
    """Variante de Zhang Eq. 3 con normalización por temperatura (sin log).

    Y_{p,j} = 1 + (N-1) / (1 + exp(-(t_{p,j} * s_{p,j}) / T))

    Aísla el aporte de la temperatura T respecto a la versión literal
    (`compute_pueblo_aspect_quality_zhang`): mantiene el conteo crudo
    t_{p,j} pero reescala la señal por T para reanclar la pendiente
    de la sigmoide y evitar saturación inmediata en pueblos con muchas
    menciones.

    Args:
        df: DataFrame con columnas ['Pueblo', 'aspect', 'sentiment'].
        T: Temperatura. Si es None, se deriva como median(|t * s| != 0).

    Returns:
        pd.DataFrame: Shape (n_pueblos, len(ALL_ASPECTS)), valores en [1, 5].
    """
    signed = df.copy()
    signed["sent_score"] = _signed_sentiment(signed) / 2.0  # → [-1, 1]

    grouped = signed.groupby(["Pueblo", "aspect"])
    s = (
        grouped["sent_score"]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(columns=ALL_ASPECTS, fill_value=0.0)
    )
    t = (
        grouped.size().unstack(fill_value=0).reindex(columns=ALL_ASPECTS, fill_value=0)
    )

    signal = t.values * s.values
    if T is None:
        nonzero_signal = np.abs(signal[signal != 0])
        T = float(np.median(nonzero_signal)) if len(nonzero_signal) > 0 else 1.0

    exponent = signal / T
    Y = pd.DataFrame(
        1.0 + (N - 1) / (1.0 + np.exp(-exponent)),
        index=s.index,
        columns=s.columns,
    )
    return Y


def compute_user_aspect_importance_zhang(df: pd.DataFrame) -> pd.DataFrame:
    """X_{u,j} fiel a Zhang et al. (2014), Eq. 2 — sigmoide sobre conteo crudo.

    t_{u,j} = número de chunks del usuario u clasificados en aspecto j.

    X_{u,j} = 1 + (N-1) * (2 / (1 + exp(-t_{u,j})) - 1)

    Equivale a la implementación actual con T = 1 (sin reescalado por
    mediana global). Útil como ablación para aislar el efecto de la
    temperatura derivada respecto al EFM original.

    Referencia:
        Zhang et al. (2014), SIGIR '14, Eq. 2.

    Returns:
        pd.DataFrame: Shape (n_users, len(ALL_ASPECTS)), valores en [1, 5].
    """
    t = (
        df.groupby(["Author", "aspect"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ALL_ASPECTS, fill_value=0)
    )
    sigmoid_shifted = 2.0 / (1.0 + np.exp(-t.values)) - 1.0
    X = pd.DataFrame(
        1.0 + (N - 1) * sigmoid_shifted,
        index=t.index,
        columns=t.columns,
    )
    return X
