from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

@dataclass
class PageRecord:
    doc_id: str
    page_number: int
    text: str

@dataclass
class ChunkRecord:
    doc_id: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    section_path: str
    text: str

def _normalize_text(s: str) -> str:
    # 温和清洗text
    # - 统一换行
    # - 去掉过多空行
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln != ""]
    return "\n".join(lines).strip()

def chunk_text_by_chars(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    """
    Return list of (start, end, chunk_text) in the ORIGINAL normalized text coordinates.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")
    
    t = text
    n = len(t)
    if n == 0:
        return []
    
    chunks: list[tuple[int, int, str]] = []
    start = 0
    step = chunk_size - overlap

    while start < n:
        end = min(start + chunk_size, n)
        chunk = t[start: end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        start += step

    return chunks

def load_pages_jsonl(pages_jsonl: Path) -> list[PageRecord]:
    pages: list[PageRecord] = []
    with pages_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pages.append(
                PageRecord(
                    doc_id=str(obj["doc_id"]),
                    page_number=int(obj["page_number"]),
                    text=str(obj.get("text", "")),
                )
            )

    return pages

def build_chunks_from_pages(
        pages: list[PageRecord],
        chunk_size: int = 1000,
        overlap: int = 150,
        min_chunk_chars: int = 80,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []

    for p in pages:
        normalized = _normalize_text(p.text or "")
        if not normalized:
            continue

        spans = chunk_text_by_chars(normalized, chunk_size=chunk_size, overlap=overlap)
        page_chunk_idx = 0
        for st, ed, ch in spans:
            if len(ch) < min_chunk_chars:
                continue
            chunks.append(
                ChunkRecord(
                    doc_id=p.doc_id,
                    page_number=p.page_number,
                    chunk_index=page_chunk_idx,
                    char_start=st,
                    char_end=ed,
                    section_path="",
                    text=ch,
                )
            )
            page_chunk_idx += 1

    return chunks

def write_chunks_jsonl(chunks: list[ChunkRecord], out_jsonl: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")