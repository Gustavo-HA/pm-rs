# Construcción de las cuatro matrices del sistema de recomendación multi-criterio.
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
    unique_reviews = (
        df.drop_duplicates(subset="review_idx")[["Author", "Pueblo", "Calificacion"]]
    )
    A = (
        unique_reviews
        .groupby(["Author", "Pueblo"])["Calificacion"]
        .mean()
        .unstack(level="Pueblo")
    )
    return A


# ---------------------------------------------------------------------------
# R_{u,p,j}  —  Sentimiento usuario–pueblo–aspecto
# ---------------------------------------------------------------------------

def compute_user_pueblo_aspect_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el sentimiento promedio por (usuario, pueblo, aspecto).

    R_{u,p,j} = mean_{s in S_{u,p,j}} sent_score(s)

    donde S_{u,p,j} son todos los chunks del usuario u sobre el pueblo p
    clasificados en el aspecto j.

    Returns:
        pd.DataFrame: MultiIndex (Author, Pueblo) x ALL_ASPECTS.
                      NaN donde el usuario no mencionó ese aspecto en ese pueblo.
    """
    signed = df.copy()
    signed["sent_score"] = _signed_sentiment(signed)

    R = (
        signed
        .groupby(["Author", "Pueblo", "aspect"])["sent_score"]
        .mean()
        .unstack(level="aspect")
        .reindex(columns=ALL_ASPECTS)
    )
    return R


# ---------------------------------------------------------------------------
# X_{u,j}  —  Importancia usuario–aspecto
# ---------------------------------------------------------------------------

def compute_user_aspect_importance(df: pd.DataFrame, T: int = 3) -> pd.DataFrame:
    """Calcula la importancia que cada usuario asigna a cada aspecto.

    t_{u,j} = número de chunks del usuario u clasificados en aspecto j
              (sobre todos los pueblos).

    X_{u,j} = 1 + (N-1) * (2 / (1 + exp(-t_{u,j} / T)) - 1)

    La función sigmoide desplazada mapea t in [0, inf) -> X in [1, N]:
      - t=0   => X=1   (aspecto nunca mencionado)
      - t->inf => X->N (aspecto muy frecuente)
      
    Args:
        df: DataFrame con columnas ['Author', 'aspect']. 
            Cada fila representa un chunk clasificado en un aspecto por un usuario.
        T: Temperatura para ajustar la sensibilidad de la función sigmoide. 
           Valores más bajos hacen que X crezca más rápido

    Returns:
        pd.DataFrame: Shape (n_users, len(ALL_ASPECTS)), valores en [1, 5].
    """
    t = (
        df.groupby(["Author", "aspect"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ALL_ASPECTS, fill_value=0)
    )
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

def compute_pueblo_aspect_quality(df: pd.DataFrame, T: int = 100) -> pd.DataFrame:
    """Calcula la calidad de cada pueblo en cada aspecto.

    h_{p,j}   = sentimiento promedio (en [-2, 2]) para pueblo p, aspecto j.
    phi_{p,j} = log(1 + |S_{p,j}|)  — volumen logarítmico de menciones.

    Y_{p,j} = 1 + (N-1) / (1 + exp(-h_{p,j} * phi_{p,j} / T))

    El producto h * phi actúa como señal volumen-ponderada:
      - Muchas menciones positivas -> Y -> N (alta calidad)
      - Pocas menciones o neutras  -> Y ~ (N+1)/2 (calidad media)
      - Menciones negativas        -> Y -> 1 (baja calidad)

    Args:
        df: DataFrame con columnas ['Pueblo', 'aspect', 'sentiment'].
             Cada fila representa un chunk clasificado en un aspecto con un sentimiento.
        T: Temperatura para ajustar la sensibilidad de la función sigmoide.

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
        grouped.size()
        .unstack(fill_value=0)
        .reindex(columns=ALL_ASPECTS, fill_value=0)
    )
    phi = np.log1p(counts.values)

    exponent = h.values * phi / T
    Y = pd.DataFrame(
        1.0 + (N - 1) / (1.0 + np.exp(-exponent)),
        index=h.index,
        columns=h.columns,
    )
    return Y
