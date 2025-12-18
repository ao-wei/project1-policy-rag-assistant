from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from policy_rag.config.settings import Settings
from policy_rag.index.chroma_store import ChromaStore
from policy_rag.retrieval.retriever import retrieve_top_k, make_snippet

console = Console()

def search(
    query: str,
    top_k: int,
    doc_id: str | None,
    category: str | None,
    show_full: bool,
):
    settings = Settings.from_repo_root()

    store = ChromaStore(
        persist_dir=settings.index_dir / "chroma",
        collection_name=settings.chroma_collection,
    )

    if store.count() == 0:
        console.print("[bold red]ERROR[/bold red] Chroma collection is empty. Run index-chunks first.")
        raise typer.Exit(code=1)
    
    # 创建搜索的筛选范围
    where = None
    if doc_id and category:
        where = {"$and": [{"doc_id": doc_id}, {"category": category}]}
    elif doc_id:
        where = {"doc_id": doc_id}
    elif category:
        where = {"category": category}

    console.print(f"\n[bold]Search[/bold] top_k={top_k}")
    console.print(f"  embedding_model: {settings.embedding_model}")
    console.print(f"  collection:      {settings.chroma_collection}")

    if where:
        console.print(f"  where:           {where}")
    console.print(f"  query:           {query}")

    hits = retrieve_top_k(
        store,
        query,
        settings.embedding_model,
        top_k,
        where,
    )

    table = Table(title="Top Retrieved Chunks (evidence candidates)", show_lines=True)
    table.add_column("Rank", justify="right")
    table.add_column("Distance", justify="right")
    table.add_column("Doc", overflow="fold")
    table.add_column("Page", justify="right")
    table.add_column("Section", overflow="fold")
    table.add_column("Snippet / Text", overflow="fold")

    for h in hits:
        md = h.metadata
        did = str(md.get("doc_id", ""))
        title = str(md.get("title", ""))
        page = str(md.get("page_number", ""))
        sec = str(md.get("section_path", "")) or ""

        content = h.text if show_full else make_snippet(h.text, max_chars=160)

        doc_show = did if not title else f"{did}\n{title}"

        table.add_row(
            str(h.rank),
            f"{h.distance:.4f}",
            doc_show,
            page,
            sec,
            content,
        )

    console.print(table)

    if hits and hits[0].distance != hits[0].distance:  # NaN
        console.print("[yellow]WARN[/yellow] Distance is NaN; check embeddings/collection config.")
    console.print(
        "\n[dim]Tip: Step 0.5 只验证“能检索到证据并带页码”。下一步我们会加 evidence gate（相似度阈值 + 不足则拒答）。[/dim]"
    )