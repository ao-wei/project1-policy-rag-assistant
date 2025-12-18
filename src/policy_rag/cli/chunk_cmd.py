from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from policy_rag.config.settings import Settings
from policy_rag.ingestion.chunking import (
    build_chunks_from_pages,
    load_pages_jsonl,
    write_chunks_jsonl,
)

console = Console()

def chunk_pages(
        doc_id: str = ...,
        chunk_size: int = 1000,
        overlap: int = 150,
        min_chunk_chars: int = 80,
):
    settings = Settings.from_repo_root()

    pages_jsonl = settings.parsed_dir / doc_id / "pages.jsonl"
    if not pages_jsonl.exists():
        console.print(f"[bold red]ERROR[/bold red] pages.jsonl not found: {pages_jsonl}")
        raise typer.Exit(code=1)
    
    pages = load_pages_jsonl(pages_jsonl)
    total_pages = len(pages)
    empty_pages = sum(1 for p in pages if not (p.text or "").strip())

    console.print(f"\n[bold]Chunking[/bold] doc_id={doc_id}")
    console.print(f"  pages: {total_pages} (empty raw pages: {empty_pages})")
    console.print(f"  chunk_size={chunk_size}, overlap={overlap}, min_chunk_chars={min_chunk_chars}")

    chunks = build_chunks_from_pages(pages, chunk_size, overlap, min_chunk_chars)

    out_jsonl = settings.parsed_dir / doc_id / "chunks.jsonl"

    write_chunks_jsonl(chunks, out_jsonl)

    console.print(f"  chunks: {len(chunks)}")
    console.print(f"[green]OK[/green] wrote: {out_jsonl}")

    if total_pages > 0 and len(chunks) == 0:
        console.print(
            "[yellow]WARN[/yellow] No chunks produced. This is often caused by scanned PDFs "
            "(no extractable text) or overly strict min_chunk_chars."
        )