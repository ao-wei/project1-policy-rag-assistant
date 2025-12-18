# 把PDF变成可用于RAG的、带页码的中间数据，负责可靠地把原始PDF内容拆成“按页文本“并落盘
# JSONL = 多个 JSON 对象按行排列
# 解决的核心问题：引用必须能对得上原文
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader

@dataclass
class ParsedPage:
    doc_id: str
    page_number: int # 1-based
    text: str

# 按页解析 PDF文件
def parse_pdf_to_pages(doc_id: str, pdf_path: Path) -> List[ParsedPage]:
    reader = PdfReader(str(pdf_path))
    pages: List[ParsedPage] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(ParsedPage(doc_id, page_number=i+1, text=text.strip()))

    return pages

# 将解析好的各页 PDF 落盘到 JSONL 文件中
def write_pages_jsonl(pages: List[ParsedPage], out_jsonl: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")