from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from rich.console import Console

from policy_rag.api.models import ChatRequest, ChatResponse, EvidenceGateInfo, RefusalPayload, Source
from policy_rag.config.settings import Settings
from policy_rag.index.chroma_store import ChromaStore
from policy_rag.llm.llm_client import ChatMessage, OllamaClient
from policy_rag.prompts.qa_prompt import SYSTEM_PROMPT, USER_TEMPLATE
from policy_rag.retrieval.evidence_gate import assess_evidence
from policy_rag.retrieval.retriever import retrieve_top_k
from policy_rag.schemas.structured_answer import StructuredAnswer
from policy_rag.utils.json_extract import extract_first_json

console = Console()
router = APIRouter()

def _build_where(doc_id: Optional[str], category: Optional[str]) -> Optional[dict[str, Any]]:
    if doc_id and category:
        return {"$and": [{"doc_id": doc_id}, {"category": category}]}
    if doc_id:
        return {"doc_id": doc_id}
    if category:
        return {"category": category}
    return None

def _format_sources_for_llm(hits, max_chars_per_source: int = 900) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        md = h.metadata or {}
        title = str(md.get("title", "") or "")
        did = str(md.get("doc_id", "") or "")
        page = md.get("page_number", "")
        sec = str(md.get("section_path", "") or "")

        text = (h.text or "").strip()
        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rstrip() + "..."
        
        header = f"[{i}] doc_id={did} title={title} page={page} section={sec}".strip()
        blocks.append(header + "\n" + text)
    return "\n\n---\n\n".join(blocks)

def _hits_to_sources(hits, max_chars: int) -> list[Source]:
    out: list[Source] = []
    for i, h in enumerate(hits, start=1):
        md = h.metadata or {}
        text = (h.text or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"

        page = md.get("page_number", None)
        try:
            page_int = int(page) if page is not None else None
        except Exception:
            page_int = None

        out.append(
            Source(
                source_id=i,
                chunk_id=h.chunk_id,
                distance=float(h.distance),
                doc_id=str(md.get("doc_id", "") or ""),
                title=str(md.get("title", "") or ""),
                category=str(md.get("category", "") or ""),
                page_number=page_int,
                section_path=str(md.get("section_path", "") or ""),
                text=text,
            )
        )       

    return out

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    settings = Settings.from_repo_root()

    store = ChromaStore(
        persist_dir=settings.index_dir / "chroma",
        collection_name=settings.chroma_collection,
    )

    if store.count() == 0:
        raise HTTPException(status_code=400, detail="Chroma collection is empty. Run ingest/index-chunks first.")
    
    where = _build_where(doc_id=req.doc_id, category=req.category)

    hits = retrieve_top_k(
        store=store,
        query=req.query,
        model_name=settings.embedding_model,
        top_k=req.top_k,
        where=where,
    )

    decision = assess_evidence(
        hits=hits,
        top1_max_dist=settings.evidence_top1_max_dist,
        good_hit_max_dist=settings.evidence_good_hit_max_dist,
        min_good_hits=settings.evidence_min_good_hits,
        min_gap=settings.evidence_min_gap,
    )

    gate_info = EvidenceGateInfo(
        ok=decision.ok,
        reasons=decision.reasons,
        suggestions=decision.suggestions,
        stats=decision.stats,
    )

    sources = _hits_to_sources(hits=hits, max_chars=req.max_chars_per_source) if req.show_sources else []

    if not decision.ok:
        refusal = RefusalPayload(
            question=req.query,
            reason="证据不足，无法给出确定性结论：" + ("；".join(decision.reasons) if decision.reasons else "未命中可靠条款"),
            follow_up_questions=decision.suggestions,
            warnings=["请以学校官方最新现行版本为准；如制度有更新/补充，请上传或指定最新文件。"],
        )
        return ChatResponse(gate=gate_info, refusal=refusal, answer=None, sources=sources)
    
    if settings.llm_provider != "ollama":
        raise HTTPException(status_code=400, detail="Only ollama provider is implemented in Step 2.1.")
    
    llm_sources = _format_sources_for_llm(hits=hits, max_chars_per_source=req.max_chars_per_source)
    user_prompt = USER_TEMPLATE.format(question=req.query, sources=llm_sources)

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
        num_predict=settings.ollama_num_predict,
    )

    raw = client.chat(
        [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]
    )

    obj = extract_first_json(raw)

    if obj.get("refusal") is True:
        refusal = RefusalPayload(
            question=req.query,
            reason=str(obj.get("reason", "模型判断证据不足，拒绝回答")),
            follow_up_questions=list(obj.get("follow_up_questions", []) or []),
            warnings=list(obj.get("warnings", []) or []) or ["请以学校官方最新现行版本为准。"],
        )
        return ChatResponse(gate=gate_info, refusal=refusal, answer=None, sources=sources)
    
    answer = StructuredAnswer.model_validate(obj)

    return ChatResponse(gate=gate_info, refusal=None, answer=answer, sources=sources)