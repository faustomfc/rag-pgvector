import logging
import sys

import click

from src.chat.loop import run_chat_loop
from src.models.embeddings import load_model
from src.persistence.engine import build_engine, initialize_database
from src.retrieval.reranker import load_reranker
from src.utils.constants import EMBEDDING_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option("--db-host",     envvar="DB_HOST",     required=True)
@click.option("--db-port",     envvar="DB_PORT",     required=True)
@click.option("--db-name",     envvar="DB_NAME",     required=True)
@click.option("--db-user",     envvar="DB_USER",     required=True)
@click.option("--db-password", envvar="DB_PASSWORD", required=True)


def main(db_host, db_port, db_name, db_user, db_password):
    logger.info("Starting RAG chat pipeline.")

    engine = build_engine(
        db_user=db_user,
        db_password=db_password,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
    )
    initialize_database(engine)

    embedding_model = load_model(EMBEDDING_MODEL)
    reranker        = load_reranker()

    run_chat_loop(engine=engine, embedding_model=embedding_model, reranker=reranker)


if __name__ == "__main__":
    main()
