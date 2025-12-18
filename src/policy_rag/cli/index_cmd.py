from __future__ import annotations

import typer
from rich.console import Console

from policy_rag.config.settings import Settings
from policy_rag.index.chroma_store import ChromaStore
from policy_rag.ingestion.indexing import load_chunks_jsonl, load_docs_meta, build_chroma_records
from policy_rag.llm.embeddings import embed_texts

console = Console()

def index_chunks(doc_id: str, batch_size: int):
    settings = Settings.from_repo_root()

    chunks_jsonl = settings.parsed_dir / doc_id / "chunks.jsonl"
    if not chunks_jsonl.exists:
        console.print(f"[bold red]ERROR[/bold red] chunks.jsonl not found: {chunks_jsonl}")
        raise typer.Exit(code=1)
    
    docs_meta = load_docs_meta(settings.docs_csv_path)
    doc_meta = docs_meta.get(doc_id)

    chunks = load_chunks_jsonl(chunks_jsonl)

    ids, documents, metadatas = build_chroma_records(doc_id, chunks, doc_meta)

    if not documents:
        console.print("[bold yellow]WARN[/bold yellow] No valid chunk texts to index.")
        raise typer.Exit(code=0)
    
    console.print(f"\n[bold]Indexing[/bold] doc_id={doc_id}")
    console.print(f"  embedding_model: {settings.embedding_model}")
    console.print(f"  chroma_dir:      {settings.index_dir / 'chroma'}")
    console.print(f"  collection:      {settings.chroma_collection}")
    console.print(f"  chunks:          {len(documents)}")

    embeddings = embed_texts(documents, model_name=settings.embedding_model, batch_size=batch_size)

    store = ChromaStore(
        persist_dir=settings.index_dir / "chroma",
        collection_name=settings.chroma_collection,
    )
    store.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    console.print(f"[green]OK[/green] upserted {len(ids)} chunks.")
    console.print(f"  collection_count_now: {store.count()}")
    
