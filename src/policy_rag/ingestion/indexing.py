# 读取 chunks.jsonl + docs.csv
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class DocMeta:
    doc_id: str
    title: str
    category: str
    file_path: str
    publish_date: str
    effective_date: str
    status: str
    source_type: str

def load_docs_meta(docs_csv_path: Path) -> dict[str, DocMeta]:
    metas: dict[str, DocMeta] = {}
    with docs_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            did = (r.get("doc_id") or "").strip()
            if not did:
                continue
            metas[did] = DocMeta(
                doc_id=did,
                title=(r.get("title") or "").strip(),
                category=(r.get("category") or "").strip(),
                file_path=(r.get("file_path") or "").strip(),
                publish_date=(r.get("publish_date") or "").strip(),
                effective_date=(r.get("effective_date") or "").strip(),
                status=(r.get("status") or "").strip(),
                source_type=(r.get("source_type") or "").strip(),
            )

    return metas

def load_chunks_jsonl(chunks_jsonl: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    with chunks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunks.append(json.loads(line))

    return chunks

def build_chroma_records(
    doc_id: str,
    chunks: list[dict[str, Any]],
    doc_meta: DocMeta | None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict[str, Any]] = []

    for c in chunks:
        page = int(c["page_number"])
        chunk_index = int(c["chunk_index"])
        char_start = int(c["char_start"])
        char_end = int(c["char_end"])
        text = str(c.get("text", "")).strip()
        section_path = str(c.get("section_path", "") or "")

        if not text:
            continue

        chunk_id = f"{doc_id}:p{page}:c{chunk_index}:{char_start}-{char_end}"
        ids.append(chunk_id)
        docs.append(text)

        md: dict[str, Any] = {
            "doc_id": doc_id,
            "page_number": page,
            "chunk_index": chunk_index,
            "char_start": char_start,
            "char_end": char_end,
            "section_path": section_path,
        }

        if doc_meta:
            md.update(
                {
                    "title": doc_meta.title,
                    "category": doc_meta.category,
                    "file_path": doc_meta.file_path,
                    "publish_date": doc_meta.publish_date,
                    "effective_date": doc_meta.effective_date,
                    "status": doc_meta.status,
                    "source_type": doc_meta.source_type,
                }
            )

        metas.append(md)

    return ids, docs, metas