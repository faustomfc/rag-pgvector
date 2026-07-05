import uuid

from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector

from src.utils.constants import EMBEDDING_DIM


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id        = Column(String,           primary_key=True, default=lambda: str(uuid.uuid4()))
    source    = Column(String,           nullable=False,   index=True)
    chunk_id  = Column(Integer,          nullable=False)
    content   = Column(Text,             nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
