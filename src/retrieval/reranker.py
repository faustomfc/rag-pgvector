import logging

import torch
from sentence_transformers import CrossEncoder

from src.utils.constants import RERANKER_MODEL, TOP_K_FINAL

logger = logging.getLogger(__name__)


def load_reranker(model_name: str = RERANKER_MODEL) -> CrossEncoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading reranker model '{model_name}' on {device}.")
    model = CrossEncoder(model_name, device=device)
    return model


def rerank(reranker: CrossEncoder, query: str, chunks: list[str], top_k: int = TOP_K_FINAL) -> list[str]:
    logger.info(f"Reranking {len(chunks)} chunks, keeping top {top_k}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs, device=device)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, _ in ranked[:top_k]]

    logger.info("Reranking complete.")
    return top_chunks
