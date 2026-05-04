# Módulo principal de inferencia para extracción de aspectos y sentimientos.
import logging
import pandas as pd
from typing import Callable
from tqdm.auto import tqdm

from transformers import (
    pipeline,
    ZeroShotClassificationPipeline,
    TextClassificationPipeline,
    RobertaForSequenceClassification,
    AutoTokenizer,
)

from .config import (
    ASPECTS_BY_TYPE,
    ASPECT_MODEL_NAME,
    SENTIMENT_MODEL_NAME,
    ID2LABEL_SENTIMENT,
    LABEL2ID_SENTIMENT,
    HYPOTHESIS_TEMPLATE,
)

logger = logging.getLogger(__name__)


class AspectSentimentExtractor:
    """
    Orquestador para la extracción conjunta de aspectos (Zero-Shot)
    y polaridad (Sequence Classification).
    """

    def __init__(self, device: str = "cuda"):
        """
        Inicializa los pipelines de Hugging Face.

        Args:
            device (str): 'cuda' para inferencia en GPU, 'cpu' en caso contrario.
        """
        logger.info(f"Inicializando modelos en dispositivo: {device}")

        # Pipeline de Aspectos (Zero-Shot)
        aspect_tokenizer = AutoTokenizer.from_pretrained(
            ASPECT_MODEL_NAME, strip_accents=False
        )
        self.aspect_classifier: ZeroShotClassificationPipeline = pipeline(
            "zero-shot-classification",
            model=ASPECT_MODEL_NAME,
            tokenizer=aspect_tokenizer,
            device=device,
        )

        # Pipeline de Sentimiento (Roberta Fine-tuned de REstMex)
        sentiment_model = RobertaForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL_NAME
        )
        sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)

        sentiment_model.config.id2label = ID2LABEL_SENTIMENT
        sentiment_model.config.label2id = LABEL2ID_SENTIMENT

        self.sentiment_classifier: TextClassificationPipeline = pipeline(
            "text-classification",
            model=sentiment_model,
            tokenizer=sentiment_tokenizer,
            device=device,
        )

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        split_function: Callable[[str], list[str]],
        batch_size: int = 64,
    ) -> pd.DataFrame:
        """
        Procesa un DataFrame completo, realizando chunking y batched inference.

        Args:
            df: DataFrame con columnas obligatorias ['Author', 'Pueblo', 'Lugar', 'Tipo', 'Calificacion', 'Review'].
            split_function: Función que toma un string y retorna una lista de strings (chunks).
            batch_size: Tamaño del lote para la inferencia en la GPU.

        Returns:
            pd.DataFrame: DataFrame expandido a nivel de chunk/sentencia con predicciones.
        """
        logger.info("Iniciando segmentación de texto...")
        records = []

        # Fusión de metadatos iterando de forma eficiente
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Splitting"):
            for chunk in split_function(row["Review"]):
                records.append(
                    {
                        "Author": row["Author"],
                        "Pueblo": row["Pueblo"],
                        "Lugar": row["Lugar"],
                        "Tipo": row["Tipo"],
                        "Calificacion": row["Calificacion"],
                        "review_idx": idx,
                        "sentence": chunk,
                    }
                )

        sent_df = pd.DataFrame(records)
        if sent_df.empty:
            logger.warning("El DataFrame resultante está vacío tras el chunking.")
            return sent_df

        logger.info(f"Total de fragmentos generados: {len(sent_df):,}")

        # Inferencia agrupada por 'Tipo' para optimizar el paso de labels al Zero-Shot
        frames = []
        for tipo, group in sent_df.groupby("Tipo"):
            labels = ASPECTS_BY_TYPE.get(tipo)
            if not labels:
                logger.warning(
                    f"No hay aspectos definidos para Tipo='{tipo}'. Omitiendo..."
                )
                continue

            sentences = group["sentence"].tolist()
            n_sentences = len(sentences)
            logger.info(f"[{tipo}] Procesando {n_sentences:,} fragmentos.")

            aspect_out, sentiment_out = [], []

            # Inferencia batcheada para Aspectos
            for i in tqdm(range(0, n_sentences, batch_size), desc=f"{tipo} - Aspectos"):
                batch = sentences[i : i + batch_size]
                res = self.aspect_classifier(
                    batch,
                    candidate_labels=labels,
                    hypothesis_template=HYPOTHESIS_TEMPLATE,
                    multi_label=True,
                    batch_size=batch_size,
                )
                aspect_out.extend(res if isinstance(res, list) else [res])

            # Inferencia batcheada para Sentimiento
            for i in tqdm(
                range(0, n_sentences, batch_size), desc=f"{tipo} - Sentimiento"
            ):
                batch = sentences[i : i + batch_size]
                res = self.sentiment_classifier(batch, top_k=1, batch_size=batch_size)
                # Manejo riguroso de la estructura de salida de text-classification
                if isinstance(res, dict):
                    res = [[res]]
                elif isinstance(res, list) and res and not isinstance(res[0], list):
                    res = [res]
                sentiment_out.extend(res)

            # Asignación de resultados vectorizada
            group = group.copy()
            expanded = []
            for row_idx, (df_idx, row) in enumerate(group.iterrows()):
                sentiment = sentiment_out[row_idx][0]
                for label, score in zip(aspect_out[row_idx]["labels"], aspect_out[row_idx]["scores"]):
                    if score >= 0.5:
                        expanded.append({
                            **row.to_dict(),
                            "aspect": label,
                            "aspect_score": score,
                            "sentiment": sentiment["label"],
                            "sentiment_score": sentiment["score"],
                        })

            if expanded:
                frames.append(pd.DataFrame(expanded))


        return pd.concat(frames, ignore_index=True)
