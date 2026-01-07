from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from policy_rag.api.models import IngestResponse
from policy_rag.config.settings import Settings
from policy_rag.index.chroma_store import ChromaStore
from policy_rag.ingestion.loader_pdf import parse_pdf_to_pages, write_pages_jsonl
from policy_rag.ingestion.chunking import build_chunks_from_pages, write_chunks_jsonl
from policy_rag.ingestion.indexing import load_docs_meta, load_chunks_jsonl, build_chroma_records
from policy_rag.llm.embeddings import embed_texts

router = APIRouter()

_DOC_ID_SAFE = re.compile(r"[^a-zA-Z0-9_\-]+")

def _sanitize_doc_id(s: str) -> str:
    s = (s or "").strip()
    s = _DOC_ID_SAFE.sub("_", s)
    s = s.strip("_")
    return s

def _gen_doc_id(title: str) -> str:
    base = _sanitize_doc_id(title)[:40] or "doc"
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}"

def _ensure_csv_header(docs_csv: Path, headers: list[str]) -> None:
    if docs_csv.exists():
        return
    docs_csv.parent.mkdir(parents=True, exist_ok=True)
    with docs_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()

def _upsert_docs_csv_row(docs_csv: Path, row: dict[str, str], headers: list[str]) -> None:
    _ensure_csv_header(docs_csv=docs_csv, headers=headers)

    rows: list[dict] = []
    found = False

    with docs_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f=f)
        for r in reader:
            if (r.get("doc_id") or "").strip() == row["doc_id"]:
                rows.append(row)
                found = True
            else:
                rows.append({h: (r.get(h) or "").strip() for h in headers})

    if not found:
        rows.append(row)

    tmp = docs_csv.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: (r.get(h) or "").strip() for h in headers})

    tmp.replace(docs_csv)

# 关键词 async，是 Python 里用来写“异步（asynchronous）代码“的语法关键字
# async def 定义的是一个协程函数，和普通 def 的区别在于：
#   普通 def：函数执行时会一直占用当前线程，知道执行完才返回
#   async def：函数内部可以在遇到等待 I/O （例如网络请求、读文件、数据库查询）的时候让出控制权，让服务器去处理别的请求
#   简单记一句：async 让“等待“不浪费线程，把时间让给别的请求
# async 必须配合 await 才有意义，await 表示愿意将执行权让出去
@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    # File、Form是FastAPI用于声明这个参数从哪里来的工具，它们告诉FastAPI：
    # “这个接口要用 multipart/form-data 解析请求体“，并把其中的不同部分（文件、表单字段）自动注入到函数参数中

    # File(...)表示这个参数来自 multipart 的文件字段（file part）
    file: UploadFile = File(..., description="Policy PDF file"),
    # Form(...)表示这个参数来自 multipart 的表单字段（key-value 项）
    doc_id: Optional[str] = Form(None),
    title: str = Form(...),
    category: str = Form(""),
    publish_date: str = Form(""),
    effective_date: str = Form(""),
    status: str = Form("in_effect"),
    source_type: str = Form("upload"),
    # ingest 参数
    reset_doc: bool = Form(False),
    chunk_size: int = Form(1000),
    overlap: int = Form(150),
    min_chunk_chars: int = Form(80),
    embed_batch_size: int = Form(32),
):
    settings = Settings.from_repo_root()

    filename = (file.filename or "").lower()
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF upload is supported for now.")
    
    did  = _sanitize_doc_id(doc_id) if doc_id else _gen_doc_id(title)
    if not did:
        raise HTTPException(status_code=400, detail="Invalid doc_id/title; cannot generate doc_id.")
    
    raw_dir = settings.repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pdf_rel_path = Path("data/raw") / f"{did}.pdf"
    pdf_abs_path = (settings.repo_root / pdf_rel_path).resolve()

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    
    pdf_abs_path.write_bytes(content)

    headers = [
        "doc_id",
        "title",
        "category",
        "publish_date",
        "effective_date",
        "status",
        "source_type",
        "file_path",
    ]

    row = {
        "doc_id": did,
        "title": title.strip(),
        "category": category.strip(),
        "publish_date": publish_date.strip(),
        "effective_date": effective_date.strip(),
        "status": status.strip(),
        "source_type": source_type.strip(),
        "file_path": str(pdf_rel_path).replace("\\", "/"),
    }

    _upsert_docs_csv_row(settings.docs_csv_path, row, headers)

    warnings: list[str] = []

    pages = parse_pdf_to_pages(doc_id, pdf_abs_path)
    pages_jsonl = settings.parsed_dir / did / "pages.jsonl"
    write_pages_jsonl(pages, pages_jsonl)

    empty_pages = sum(1 for p in pages if not (p.text or "").strip())
    if len(pages) > 0 and empty_pages / len(pages) >= 0.6:
        warnings.append("PDF 可能为扫描件（可提取文字较少）。后续可能需要 OCR 才能稳定检索。")

    chunks = build_chunks_from_pages(
        pages, 
        chunk_size=int(chunk_size), 
        overlap=int(overlap), 
        min_chunk_chars=int(min_chunk_chars),
    )
    chunks_jsonl = settings.parsed_dir / did / "chunks.jsonl"
    write_chunks_jsonl(chunks, chunks_jsonl)

    docs_meta = load_docs_meta(settings.docs_csv_path)
    meta = docs_meta.get(did)
    chunks_raw = load_chunks_jsonl(chunks_jsonl)
    ids, documents, metadatas = build_chroma_records(
        doc_id=doc_id,
        chunks=chunks_raw,
        doc_meta=meta,
    )

    if not documents:
        warnings.append("未生成可用 chunk（可能是扫描件或 min_chunk_chars 过大），已跳过向量入库。")
        store = ChromaStore(
            persist_dir=settings.index_dir / "chroma",
            collection_name=settings.chroma_collection,
        )
        return IngestResponse(
            doc_id=did,
            file_path=row["file_path"],
            pages=len(pages),
            empty_pages=empty_pages,
            chunks=len(chunks),
            indexed_chunks=0,
            collection_count_now=store.count(),
            warnings=warnings,
        )
    
    store = ChromaStore(
        persist_dir=settings.index_dir / "chroma",
        collection_name=settings.chroma_collection,
    )

    if reset_doc:
        store.delete(where={"doc_id": doc_id})

    embeddings = embed_texts(documents, model_name=settings.embedding_model, batch_size=embed_batch_size)
    store.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    return IngestResponse(
        doc_id=did,
        file_path=row["file_path"],
        pages=len(pages),
        empty_pages=empty_pages,
        chunks=len(chunks),
        indexed_chunks=len(ids),
        collection_count_now=store.count(),
        warnings=warnings,
    )
