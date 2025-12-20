from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from policy_rag.config.settings import Settings
from policy_rag.index.chroma_store import ChromaStore
from policy_rag.retrieval.retriever import retrieve_top_k, make_snippet
from policy_rag.retrieval.evidence_gate import assess_evidence

console = Console()

def search(
    query: str,
    top_k: int,
    doc_id: str | None,
    category: str | None,
    show_full: bool,
    use_gate: bool = True,
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

    decision = None
    if use_gate:
        decision = assess_evidence(
            hits=hits,
            top1_max_dist=settings.evidence_top1_max_dist,
            good_hit_max_dist=settings.evidence_good_hit_max_dist,
            min_good_hits=settings.evidence_min_good_hits,
            min_gap=settings.evidence_min_gap,
        )

        if decision.ok:
            console.print("[bold green]EVIDENCE: OK[/bold green] 证据充足，可以进入回答阶段（后续 Step 1.x 会接 LLM）")
        else:
            console.print("[bold yellow]EVIDENCE: INSUFFICIENT[/bold yellow] 证据不足：当前不建议生成确定性结论")
            for r in decision.reasons:
                console.print(f"  - {r}")
            if decision.suggestions:
                console.print("\n[bold]建议你补充/改写问题：[/bold]")
                for s in decision.suggestions:
                    console.print(f"  - {s}")
            console.print(f"\n[dim]gate stats: {decision.stats}[/dim]\n")
        

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
        "\n[dim]Tip: 证据不足时，我们会“拒绝给出确定结论 + 给出追问建议”，这是后续 ask/summarize 强制执行的反幻觉策略。[/dim]"
    )