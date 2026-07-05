import logging

from sentence_transformers import SentenceTransformer
from sqlalchemy.engine import Engine

from sentence_transformers import CrossEncoder
from src.chat.history import ConversationHistory
from src.generation.llm import generate_answer
from src.retrieval.query import rewrite_query
from src.retrieval.reranker import rerank
from src.retrieval.search import search_chunks

logger = logging.getLogger(__name__)

_QUIT_COMMAND = "/quit"


def run_chat_loop(
    engine: Engine,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
) -> None:
    history = ConversationHistory()

    print("\n===================================")
    print("RAG CHAT STARTED")
    print(f"Type {_QUIT_COMMAND} to exit")
    print("===================================\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == _QUIT_COMMAND:
            logger.info("User requested exit.")
            print("\nGoodbye.")
            break

        if not question:
            continue

        logger.info(f"Received question: {question}")

        contextual_query = rewrite_query(question, history.get())

        query_embedding = embedding_model.encode(
            [contextual_query],
            normalize_embeddings=True,
        )[0].tolist()

        candidates = search_chunks(engine, query_embedding)
        top_chunks  = rerank(reranker, contextual_query, candidates)
        answer      = generate_answer(question, top_chunks, history.get())

        print(f"\nAssistant:\n{answer}\n")

        history.add(question, answer)
        logger.info("Turn complete.")
