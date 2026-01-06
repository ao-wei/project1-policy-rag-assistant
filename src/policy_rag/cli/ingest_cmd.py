from __future__ import annotations

import csv
from pathlib import Path

import typer
from rich.console import Console

from policy_rag.config.settings import Settings
from policy_rag.ingestion.validators import validate_docs_csv
from policy_rag.ingestion.loader_pdf import parse_pdf_to_pages, write_pages_jsonl
from policy_rag.ingestion.chunking import load_pages_jsonl, build_chunks_from_pages, write_chunks_jsonl
from policy_rag.ingestion.indexing import load_docs_meta, load_chunks_jsonl, build_chroma_records
from policy_rag.llm.embeddings import embed_texts
from policy_rag.index.chroma_store import ChromaStore

console = Console()

def _load_docs_rows(docs_csv_path: Path) -> list[dict]:
    with docs_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
    
def ingest(
    doc_id: str | None = None,
    all_docs: bool = False,
    reparse: bool = False,
    rechunk: bool = False,
    reset_doc: bool = False,
    chunk_size: int = 1000,
    overlap: int = 150,
    min_chunk_chars: int = 80,
    embed_batch_size: int = 32,
):
    """
    One-shot ingest pipeline:
    docs.csv -> PDF parse (pages) -> chunk -> embed -> upsert to Chroma
    """
    settings = Settings.from_repo_root()

    issues = validate_docs_csv(settings.docs_csv_path, settings.repo_root)
    errors = [i for i in issues if i.level == "ERROR"]
    if errors:
        console.print("[bold red]ERROR[/bold red] docs.csv validation failed. Run `policy-rag validate-metadata` to see details.")
        raise typer.Exit(code=1)
    
    rows = _load_docs_rows(settings.docs_csv_path)
    if not rows:
        console.print("[bold red]ERROR[/bold red] docs.csv is empty.")
        raise typer.Exit(code=1)
    
    if not all_docs and not doc_id:
        console.print("[bold red]ERROR[/bold red] Provide --doc-id or use --all-docs.")
        raise typer.Exit(code=1)
    
    target_rows: list[dict]
    if all_docs:
        target_rows = rows
    else:
        target_rows = [r for r in rows if (r.get("doc_id") or "").strip() == doc_id]
        if not target_rows:
            console.print(f"[bold red]ERROR[/bold red] doc_id not found in docs.csv: {doc_id}")
            raise typer.Exit(code=1)
        
    docs_meta = load_docs_meta(settings.docs_csv_path)

    store = ChromaStore(
        persist_dir=settings.index_dir / "chroma",
        collection_name=settings.chroma_collection,
    )

    console.print("\n[bold]Ingest Pipeline[/bold]")
    console.print(f"  embedding_model: {settings.embedding_model}")
    console.print(f"  chroma_dir:      {settings.index_dir / 'chroma'}")
    console.print(f"  collection:      {settings.chroma_collection}")
    console.print(f"  reparse={reparse}, rechunk={rechunk}, reset_doc={reset_doc}")
    console.print(f"  chunk_size={chunk_size}, overlap={overlap}, min_chunk_chars={min_chunk_chars}")
    console.print(f"  embed_batch_size={embed_batch_size}")

    for r in target_rows:
        did = (r.get("doc_id") or "").strip()
        file_path = (r.get("file_path") or "").strip()

        meta = docs_meta.get(did)
        title = meta.title if meta else (r.get("title") or "").strip()

        pdf_path = (settings.repo_root / file_path).resolve()

        if not pdf_path.exists():
            console.print(f"\n[bold red]SKIP[/bold red] doc_id={did} PDF not found: {file_path}")
            continue

        console.print(f"\n[bold]Doc[/bold] doc_id={did}")
        if title:
            console.print(f"  title: {title}")
        console.print(f"  pdf:   {pdf_path}")

        pages_jsonl = settings.parsed_dir / did / "pages.jsonl"
        if reparse or (not pages_jsonl.exists()):
            pages = parse_pdf_to_pages(did, pdf_path)
            write_pages_jsonl(pages, pages_jsonl)
            empty_pages = sum(1 for p in pages if not (p.text or "").strip())
            console.print(f"  parse: wrote pages.jsonl ({len(pages)} pages, empty={empty_pages})")
            if len(pages) > 0 and empty_pages / len(pages) >= 0.6:
                console.print("[yellow]  WARN[/yellow] Many pages empty; may be scanned PDF (OCR later).")
        else:
            console.print("  parse: skip (pages.jsonl exists)")

        chunks_jsonl = settings.parsed_dir / did / "chunks.jsonl"
        if rechunk or (not chunks_jsonl.exists()):
            pages = load_pages_jsonl(pages_jsonl)
            chunks = build_chunks_from_pages(
                pages,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_chars=min_chunk_chars,
            )
            write_chunks_jsonl(chunks, chunks_jsonl)
            console.print(f"  chunk: wrote chunks.jsonl ({len(chunks)} chunks)")
        else:
            console.print("  chunk: skip (chunks.jsonl exists)")

        chunks_raw =load_chunks_jsonl(chunks_jsonl)
        ids, documents, metadatas = build_chroma_records(did, chunks_raw, meta)

        if not documents:
            console.print("[bold yellow]  WARN[/bold yellow] No valid chunk texts. Skipping indexing.")
            continue

        if reset_doc:
            store.delete(where={"doc_id": did})
            console.print("  index: cleared existing vectors for this doc_id")

        embeddings = embed_texts(documents, model_name=settings.embedding_model, batch_size=embed_batch_size)
        store.upsert(ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

        console.print(f"  index: upserted {len(ids)} chunks")
        console.print(f"  index: collection_count_now={store.count()}")

    console.print("\n[bold green]DONE[/bold green] ingest finished.")