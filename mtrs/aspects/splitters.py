# Estrategias de segmentación de texto (chunking).
import nltk
from nltk.tokenize import sent_tokenize

# Asegurar la descarga del modelo de puntuación
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def split_review_punkt(text: str, max_len: int = 400, min_len=10) -> list[str]:
    """
    Segmenta un texto en oraciones utilizando el tokenizador de NLTK.
    Si una oración excede `max_len`, realiza un fallback a segmentación por caracteres.
    """
    sentences = sent_tokenize(text, language="spanish")
    result = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) <= min_len:
            continue
        if len(s) <= max_len:
            result.append(s)
        else:
            # Fallback a split por caracteres para sentencias atípicamente largas
            for i in range(0, len(s), max_len):
                sub = s[i : i + max_len].strip()
                if sub:
                    result.append(sub)

    return result


def split_review_char(text: str, chunk_size: int = 100) -> list[str]:
    """
    Segmenta un texto en bloques de caracteres estrictos de tamaño `chunk_size`.
    """
    return [
        text[i : i + chunk_size].strip()
        for i in range(0, len(text), chunk_size)
        if text[i : i + chunk_size].strip()
    ]
