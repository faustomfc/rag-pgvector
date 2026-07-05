import logging
import os
import re
import uuid

import fitz
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.persistence.schema import DocumentChunk
from src.utils.constants import BATCH_SIZE, CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_LEN

logger = logging.getLogger(__name__)


def _generate_chunks(text: str, chunk_size: int, chunk_overlap: int, min_chunk_len: int) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += " " + sentence
        else:
            if current.strip():
                chunks.append(current.strip())
            current = current[-chunk_overlap:] + " " + sentence

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) >= min_chunk_len]


def _extract_chunks_from_pdf(path: str, filename: str) -> list[dict]:
    doc = fitz.open(path)
    raw = "".join(page.get_text() for page in doc)
    doc.close()

    raw = raw.replace("\x00", "")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()

    chunks = _generate_chunks(raw, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, min_chunk_len=MIN_CHUNK_LEN)
    logger.info(f"{filename}: {len(chunks)} chunks generated.")

    return [
        {"id": str(uuid.uuid4()), "source": filename, "chunk_id": idx, "content": chunk}
        for idx, chunk in enumerate(chunks)
    ]


def load_pdfs(folder: str, engine: Engine) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT source FROM document_chunks"))
        already_indexed = {row[0] for row in result}

    files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    new_files = [f for f in files if f not in already_indexed]

    if not new_files:
        logger.info("No new PDFs found.")
        return []

    logger.info(f"{len(new_files)} new PDFs out of {len(files)} found.")

    all_chunks = []
    for filename in new_files:
        try:
            chunks = _extract_chunks_from_pdf(os.path.join(folder, filename), filename)
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.error(f"Failed to process '{filename}': {exc}")

    logger.info(f"Total chunks extracted: {len(all_chunks)}")
    return all_chunks


def index_chunks(engine: Engine, chunks: list[dict], embeddings: list[list[float]], batch_size: int = BATCH_SIZE) -> None:
    records = [
        {"id": c["id"], "source": c["source"], "chunk_id": c["chunk_id"], "content": c["content"], "embedding": emb}
        for c, emb in zip(chunks, embeddings)
    ]

    logger.info(f"Indexing {len(records)} chunks in batches of {batch_size}.")

    with Session(engine) as session:
        for start in tqdm(range(0, len(records), batch_size)):
            batch = records[start:start + batch_size]
            stmt = insert(DocumentChunk).values(batch).on_conflict_do_nothing(index_elements=["id"])
            session.execute(stmt)
            session.commit()

    logger.info(f"{len(records)} chunks indexed successfully.")


def create_hnsw_index(engine: Engine) -> None:
    logger.info("Creating HNSW index on embedding column.")
    sql = text("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
        ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    with engine.connect() as conn:
        conn.execute(sql)
        conn.commit()
    logger.info("HNSW index created successfully.")
