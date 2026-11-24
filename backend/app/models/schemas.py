
from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    language: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool
    question: str
    answer: str
    detected_language: str
    source: str
    response_time: float
    timestamp: str
