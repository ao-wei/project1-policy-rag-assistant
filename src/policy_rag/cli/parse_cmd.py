# 把doc.csv 里登记的 PDF 文档按“doc_id"取出来，逐页解析成文本，并把每一页（带页码）写成 pages.jsonl 落盘，顺便在终端打印解析统计
from __future__ import annotations

import csv
from pathlib import Path

import typer
from rich.console import Console

from policy_rag.config.settings import Settings
from policy_rag.ingestion.loader_pdf import parse_pdf_to_pages, write_pages_jsonl

console = Console()

def _load_docs_rows(docs_csv_path: Path) -> list[dict]:
    with docs_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
    
# = typer.Option(...)是 Typer 框架的写法：它用函数参数来声明命令行参数 CLI options
def parse_pdf(doc_id: str | None = None,
              all_docs: bool = False
              ):
    settings = Settings.from_repo_root()
    rows = _load_docs_rows(settings.docs_csv_path)

    if not rows:
        console.print("[bold red]ERROR[/bold red] docs.csv is empty.")
        raise typer.Exit(code=1)
    
    target_rows: list[dict]
    if all_docs:
        target_rows = rows
    else:
        if not doc_id:
            console.print("[bold red]ERROR[/bold red] Provide --doc-id or use --all-docs.")
            raise typer.Exit(code=1)
        target_rows = [r for r in rows if (r.get("doc_id") or "").strip() == doc_id]
        if not target_rows:
            console.print(f"[bold red]ERROR[/bold red] doc_id not found in docs.csv: {doc_id}")
            raise typer.Exit(code=1)
        
    for r in target_rows:
        did = (r.get("doc_id") or "").strip()
        file_path = (r.get("file_path") or "").strip()
        title = (r.get("title") or "").strip()

        pdf_path = (settings.repo_root / file_path).resolve()
        if not pdf_path.exists():
            console.print(f"[bold red]ERROR[/bold red] PDF not found: {file_path}")
            raise typer.Exit(code=1)
        
        console.print(f"\n[bold]Parsing[/bold] doc_id={did}")
        if title:
            console.print(f"  title: {title}")
        console.print(f"  pdf:   {pdf_path}")

        pages = parse_pdf_to_pages(did, pdf_path)
        out_jsonl = settings.parsed_dir / did / "pages.jsonl"
        write_pages_jsonl(pages, out_jsonl)

        empty_pages = sum(1 for p in pages if len(p.text.strip()) == 0)
        console.print(f"  pages: {len(pages)}")
        console.print(f"  empty: {empty_pages}")

        if len(pages) > 0 and empty_pages / len(pages) >= 0.6:
            console.print(
                "[yellow]WARN[/yellow] Many pages are empty. This PDF may be scanned/image-based "
                "and may need OCR later."
            )

        console.print(f"[green]OK[/green] wrote: {out_jsonl}")