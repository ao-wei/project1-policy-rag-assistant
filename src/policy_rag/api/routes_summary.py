from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from policy_rag.api.models import DocSummaryResponse, RefusalPayload, Source
from policy_rag.config.settings import Settings
from policy_rag.index.chroma_store import ChromaStore
from policy_rag.ingestion.indexing import load_docs_meta
from policy_rag.llm.llm_client import OllamaClient, ChatMessage
from policy_rag.prompts.policy_card_prompt import SYSTEM_PROMPT, USER_TEMPLATE
from policy_rag.schemas.structured_answer import StructuredAnswer
from policy_rag.utils.json_extract import extract_first_json

router = APIRouter()

def _pick_representative_sources(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    max_sources: int = 32,
) -> list[dict]:
    by_page: dict[int, dict] = {}
    fallback: list[dict] = []

    for cid, doc, md in zip(ids, documents, metadatas):
        text = str(doc or "").strip()
        if not text:
            continue
        md = md or {}

        fallback.append({"chunk_id": cid, "text": text, "md": md})

        page = md.get("page_number", None)
        try:
            page_int = int(page) if page is not None else None
        except Exception:
            page_int = None

        if page_int is None:
            continue

        cur = by_page.get(page_int)
        if cur is None or len(text) > len(cur["text"]):
            by_page[page_int] = {"chunk_id": cid, "text": text, "md": md}

    picked: list[dict] = []
    if by_page:
        for p in sorted(by_page.keys()):
            picked.append(by_page[p])
            if len(picked) >= max_sources:
                break
        return picked
    
    return fallback[:max_sources]

def _format_sources_for_llm(picked: list[dict], max_chars_per_source: int = 900) -> str:
    blocks = []
    for i, it in enumerate(picked, start=1):
        md = it["md"] or {}
        did = str(md.get("doc_id", "") or "")
        title = str(md.get("title", "") or "")
        page = md.get("page_number", "")
        sec = str(md.get("section_path", "") or "")

        text = (it["text"] or "").strip()
        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rstrip() + "…"

        header = f"[{i}] doc_id={did} title={title} page={page} section={sec}".strip()
        blocks.append(header + "\n" + text)
    return "\n\n---\n\n".join(blocks)

def _picked_to_sources(picked: list[dict], max_char: int) -> list[Source]:
    out: list[Source] = []
    for i, it in enumerate(picked, start=1):
        md = it["md"] or {}
        text = (it["text"] or "").strip()
        if len(text) > max_char:
            text = text[:max_char].rstrip() + "..."

        page = md.get("page_number", None)
        try:
            page_int = int(page) if page is not None else None
        except Exception:
            page_int = None

        out.append(
            Source(
                source_id=i,
                chunk_id=str(it.get("chunk_id", "")),
                distance=0.0,
                doc_id=str(md.get("doc_id", "") or ""),
                title=str(md.get("title", "") or ""),
                category=str(md.get("category", "") or ""),
                page_number=page_int,
                section_path=str(md.get("section_path", "") or ""),
                text=text,
            )
        )
    return out

@router.get("/doc/{doc_id}/summary", response_model=DocSummaryResponse)
def doc_summary(
    doc_id: str,
    max_sources: int = Query(32, ge=4, le=40),
    show_sources: bool = Query(True),
    max_chars_per_source: int = Query(1800, ge=200, le=2000)
) -> DocSummaryResponse:
    settings = Settings.from_repo_root()

    docs_meta = load_docs_meta(settings.docs_csv_path)
    meta = docs_meta.get(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"doc_id not found in docs.csv: {doc_id}")
    
    store = ChromaStore(
        persist_dir=settings.index_dir / "chroma",
        collection_name=settings.chroma_collection,
    )
    if store.count() == 0:
        raise HTTPException(status_code=400, detail="Chroma collection is empty. Run ingest first.")
    
    got = store.get(where={"doc_id": doc_id}, limit=5000)
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []

    if not docs:
        raise HTTPException(status_code=404, detail=f"No chunks found for doc_id={doc_id}. Did you ingest/index it?")
    
    picked = _pick_representative_sources(ids=ids, documents=docs, metadatas=metas, max_sources=max_sources)

    sources = _picked_to_sources(picked=picked, max_char=max_chars_per_source) if show_sources else []

    if settings.llm_provider != "ollama":
        raise HTTPException(status_code=400, detail="Only ollama provider is implemented in Step 2.3.")
    
    llm_sources = _format_sources_for_llm(picked, max_chars_per_source)

    user_prompt = USER_TEMPLATE.format(
        doc_id=meta.doc_id,
        title=meta.title,
        category=meta.category,
        publish_date=meta.publish_date,
        effective_date=meta.effective_date,
        status=meta.status,
        sources=llm_sources,
    )

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
            question=str(obj.get("question", meta.title)),
            reason=str(obj.get("reason", "模型判断证据不足，拒绝总结")),
            follow_up_questions=list(obj.get("follow_up_questions", []) or []),
            warnings=list(obj.get("warnings", []) or []) or ["请以学校官方最新现行版本为准。"],
        )
        return DocSummaryResponse(
            doc_id=meta.doc_id,
            title=meta.title,
            category=meta.category,
            publish_date=meta.publish_date,
            effective_date=meta.effective_date,
            status=meta.status,
            refusal=refusal,
            summary=None,
            sources=sources,
            warnings=refusal.warnings,
        )
    
    summary = StructuredAnswer.model_validate(obj)

    warnings = list(summary.warnings or [])
    if not any("最新" in w or "现行" in w for w in warnings):
        warnings.append("请以学校官方最新现行版本为准；如制度更新，请上传/指定最新文件。")

    summary.warnings = warnings

    return DocSummaryResponse(
        doc_id=meta.doc_id,
        title=meta.title,
        category=meta.category,
        publish_date=meta.publish_date,
        effective_date=meta.effective_date,
        status=meta.status,
        refusal=None,
        summary=summary,
        sources=sources,
        warnings=warnings,
    )