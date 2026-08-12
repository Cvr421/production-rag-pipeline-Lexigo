from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    include_trace: bool = False


class Citation(BaseModel):
    chunk_id: str
    source_file: str
    chunk_index: int
    score: float = Field(ge=-1.0, le=1.0)
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[Citation]
    trace: list[dict[str, Any]] | None = None

