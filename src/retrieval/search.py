import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.utils.constants import TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)


def search_chunks(engine: Engine, query_embedding: list[float], top_k: int = TOP_K_RETRIEVAL) -> list[str]:
    logger.info(f"Searching top {top_k} chunks by cosine similarity.")

    sql = text("""
        SELECT content
        FROM document_chunks
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    with engine.connect() as conn:
        result = conn.execute(sql, {"embedding": str(query_embedding), "top_k": top_k})
        chunks = [row[0] for row in result]

    logger.info(f"{len(chunks)} chunks retrieved.")
    return chunks
