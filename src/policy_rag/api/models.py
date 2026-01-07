# API 请求/响应模型
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from policy_rag.schemas.structured_answer import StructuredAnswer

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(8, ge=1, le=30)
    doc_id: Optional[str] = None
    category: Optional[str] = None
    show_sources: bool = True
    max_chars_per_source: int = Field(900, ge=200, le=2000)

class EvidenceGateInfo(BaseModel):
    ok: bool
    reasons: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)

class Source(BaseModel):
    source_id: int
    chunk_id: str
    distance: float
    doc_id: str
    title: str = ""
    category: str = ""
    page_number: Optional[int] = None
    section_path: str = ""
    text: str

class RefusalPayload(BaseModel):
    refusal: bool = True
    question: str
    reason: str
    follow_up_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class ChatResponse(BaseModel):
    gate: EvidenceGateInfo
    refusal: Optional[RefusalPayload] = None
    answer: Optional[StructuredAnswer] = None
    sources: list[Source] = Field(default_factory=list)

