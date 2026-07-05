import logging
import sys

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from sqlalchemy import text

from src.api.schemas import ChatRequest, ChatResponse, DocumentInfo, HealthResponse
from src.api.session import SessionManager
from src.generation.llm import generate_answer
from src.models.embeddings import load_model
from src.persistence.engine import build_engine, initialize_database
from src.retrieval.query import rewrite_query
from src.retrieval.reranker import load_reranker, rerank
from src.retrieval.search import search_chunks
from src.utils.constants import EMBEDDING_MODEL

import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

engine         = None
embedding_model = None
reranker       = None
session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, embedding_model, reranker

    logger.info("Starting RAG API.")

    engine = build_engine(
        db_user=os.environ["DB_USER"],
        db_password=os.environ["DB_PASSWORD"],
        db_host=os.environ["DB_HOST"],
        db_port=os.environ["DB_PORT"],
        db_name=os.environ["DB_NAME"],
    )
    initialize_database(engine)

    embedding_model = load_model(EMBEDDING_MODEL)
    reranker        = load_reranker()

    logger.info("RAG API ready.")
    yield
    logger.info("Shutting down RAG API.")


app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation over technical documents.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"

    return HealthResponse(status="ok", database=db_status)


@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT source, COUNT(*) as chunk_count
                FROM document_chunks
                GROUP BY source
                ORDER BY source
            """))
            return [
                DocumentInfo(source=row[0], chunk_count=row[1])
                for row in result
            ]
    except Exception as exc:
        logger.error(f"Failed to fetch documents: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch documents.")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(f"[{request.session_id}] Question: {request.question}")

    history = session_manager.get_or_create(request.session_id)

    contextual_query = rewrite_query(request.question, history.get())

    query_embedding = embedding_model.encode(
        [contextual_query],
        normalize_embeddings=True,
    )[0].tolist()

    candidates  = search_chunks(engine, query_embedding)
    top_chunks  = rerank(reranker, contextual_query, candidates)
    answer      = generate_answer(request.question, top_chunks, history.get())

    history.add(request.question, answer)

    logger.info(f"[{request.session_id}] Answer generated.")

    return ChatResponse(
        session_id=request.session_id,
        question=request.question,
        answer=answer,
    )
