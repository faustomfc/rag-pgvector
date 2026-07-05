from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str


class DocumentInfo(BaseModel):
    source: str
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    database: str
