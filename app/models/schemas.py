from typing import Optional

from pydantic import BaseModel


class SearchRequest(BaseModel):
    question: str
    top_k: int = 3


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class AgentChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
