# Configuraciones y constantes para la extracción de aspectos y sentimientos.

ASPECTS_BY_TYPE: dict[str, list[str]] = {
    "Restaurant": [
        "servicio",
        "precio",
        "ambiente",
        "comida",
    ],
    "Hotel": [
        "servicio",
        "precio",
        "ambiente",
        "ubicación",
        "habitación",
    ],
    "Attractive": [
        "servicio",
        "ambiente",
        "precio",
        "naturaleza",
        "cultura",
    ],
}

# Modelos en Hugging Face
ASPECT_MODEL_NAME = "Recognai/zeroshot_selectra_medium"
SENTIMENT_MODEL_NAME = "vg055/roberta-base-bne-finetuned-TripAdvisorDomainAdaptation-finetuned-e2-RestMex2023-polaridadDA-V1"

# Mapeo de etiquetas para el modelo de sentimiento
ID2LABEL_SENTIMENT = {
    0: "Muy Negativo",
    1: "Negativo",
    2: "Neutral",
    3: "Positivo",
    4: "Muy Positivo"
}
LABEL2ID_SENTIMENT = {v: int(k) for k, v in ID2LABEL_SENTIMENT.items()}

HYPOTHESIS_TEMPLATE = "Este texto trata sobre {}."