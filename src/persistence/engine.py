import logging

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.persistence.schema import Base

logger = logging.getLogger(__name__)


def build_engine(db_user: str, db_password: str, db_host: str, db_port: str, db_name: str) -> Engine:
    url = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    logger.info(f"Connecting to database '{db_name}' at {db_host}:{db_port}.")
    return create_engine(url, echo=False, pool_pre_ping=True)


def initialize_database(engine: Engine) -> None:
    logger.info("Initializing database.")
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    logger.info("Database initialized.")
