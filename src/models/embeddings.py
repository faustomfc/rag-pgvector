import logging

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def load_model(embedding_model: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    logger.info(f"Loading embedding model '{embedding_model}' on {device}.")
    return SentenceTransformer(embedding_model, device=device, model_kwargs=kwargs or None)


def generate_embeddings(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    logger.info(f"Generating embeddings for {len(texts)} chunks.")
    vectors = model.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    return [v.tolist() for v in vectors]
