from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from policy_rag.config.settings import Settings
from policy_rag.index.chroma_store import ChromaStore
from policy_rag.retrieval.retriever import retrieve_top_k, make_snippet
from policy_rag.retrieval.evidence_gate import assess_evidence
from policy_rag.llm.llm_client import OllamaClient, ChatMessage
from policy_rag.prompts.qa_prompt import SYSTEM_PROMPT, USER_TEMPLATE
from policy_rag.utils.json_extract import extract_first_json
from policy_rag.schemas.answer import Refusal
from policy_rag.schemas.structured_answer import StructuredAnswer

console = Console()

def _format_sources_for_llm(hits, max_chars_per_source: int = 1000) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        md = h.metadata or {}
        title = str(md.get("title", "") or "")
        did = str(md.get("doc_id", "") or "")
        page = md.get("page_number", "")
        sec = str(md.get("section_path", "") or "")

        text = (h.text or "").strip()
        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rstrip() + "..."

        header = f"[{i}] doc_id={did} title={title} page={page} section={sec}".strip()
        blocks.append(header + "\n" + text)

    return "\n\n---\n\n".join(blocks)

def _print_evidence_table(hits):
    table = Table(title="Top Retrieved Chunks (evidence)", show_lines=True)
    table.add_column("Rank", justify="right")
    table.add_column("Distance", justify="right")
    table.add_column("Doc", overflow="fold")
    table.add_column("Page", justify="right")
    table.add_column("Snippet", overflow="fold")

    for h in hits:
        md = h.metadata or {}
        did = str(md.get("doc_id", ""))
        title = str(md.get("title", ""))
        page = str(md.get("page_number", ""))
        doc_show = did if not title else f"{did}\n{title}"
        table.add_row(str(h.rank), f"{h.distance:.4f}", doc_show, page, make_snippet(h.text, 160))
    console.print(table)

# 将LLM 生成的“结构化字段（list[Items]）打印成用户可读的分组答案，并把每条要点的引用（页码+原文摘录）补全展示出来
def _render_items(title: str, items, hits):
    if not items:
        return 
    console.print(f"\n[bold]{title}[/bold]")
    for i, it in enumerate(items, start=1):
        console.print(f"{i}. {it.text}  [dim]({it.confidence})[/dim]")
        for cit in it.citations:
            sid = cit.source_id
            if sid < 1 or sid > len(hits):
                console.print(f"   - [red]Citation error[/red]: source_id={sid} out of range")
                continue
            h = hits[sid - 1]
            md = h.metadata or {}
            title0 = str(md.get("title", "") or "")
            did = str(md.get("doc_id", "") or "")
            page = md.get("page_number", "")
            quote = cit.quote.strip().replace("\n", " ")
            if len(quote) > 120:
                quote = quote[:120].rstrip() + "…"
            console.print(f"   - 引用：{title0 or did} | doc_id={did} | p.{page} | “{quote}”")

def ask(
    query: str,
    top_k: int,
    doc_id: str | None,
    category: str | None,
    use_gate: bool,
    show_evidence: bool
):
    settings = Settings.from_repo_root()

    store = ChromaStore(
        persist_dir=settings.index_dir / "chroma",
        collection_name=settings.chroma_collection,
    )

    if store.count() == 0:
        console.print("[bold red]ERROR[/bold red] Chroma collection is empty. Run index-chunks first.")
        raise typer.Exit(code=1)
    
    where = None
    if doc_id and category:
        where = {"$and": [{"doc_id": doc_id}, {"category": category}]}
    elif doc_id:
        where = {"doc_id": doc_id}
    elif category:
        where = {"category": category}

    hits = retrieve_top_k(
        store=store,
        query=query,
        model_name=settings.embedding_model,
        top_k=top_k,
        where=where,
    )

    if show_evidence:
        _print_evidence_table(hits=hits)

    if use_gate:
        decision = assess_evidence(
            hits=hits,
            top1_max_dist=settings.evidence_top1_max_dist,
            good_hit_max_dist=settings.evidence_good_hit_max_dist,
            min_good_hits=settings.evidence_min_good_hits,
            min_gap=settings.evidence_min_gap,
        )

        if not decision.ok:
            console.print("\n[bold yellow]结论：证据不足（拒绝给出确定回答）[/bold yellow]")
            for r in decision.reasons:
                console.print(f"  - {r}")
            if decision.suggestions:
                console.print("\n[bold]你可以这样补充信息/改写问题：[/bold]")
                for s in decision.suggestions:
                    console.print(f"  - {s}")

            console.print(f"\n[dim]gate stats: {decision.stats}[/dim]")
            raise typer.Exit(code=0)
        
    if settings.llm_provider != "ollama":
        console.print("[bold red]ERROR[/bold red] Step 1.1 只实现 ollama provider。请设置 LLM_PROVIDER=ollama。")
        raise typer.Exit(code=1)
    
    source_str = _format_sources_for_llm(hits, max_chars_per_source=1000)
    user_prompt = USER_TEMPLATE.format(question=query, sources=source_str)

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
        num_predict=settings.ollama_num_predict,
    )

    console.print(f"\n[bold]LLM[/bold] provider=ollama model={settings.ollama_model}")

    raw = client.chat(
        [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ],
        response_format=StructuredAnswer.model_json_schema(),
    )

    obj = extract_first_json(raw)

    # schema validate (AskAnswer or Refusal)
    parsed = None
    if obj.get("refusal") is True:
        # 把obj 按 Refusal 这个“结构合同“做严格校验，并转换成一个类型安全的 Refusal 实例
        parsed = Refusal.model_validate(obj=obj)
        console.print("\n[bold yellow]模型拒答（Refusal）[/bold yellow]")
        console.print(f"- 原因：{parsed.reason}")
        if parsed.follow_up_questions:
            console.print("- 需要补充：")
            for q in parsed.follow_up_questions:
                console.print(f"  - {q}")
        if parsed.warnings:
            console.print("- 提醒：")
            for w in parsed.warnings:
                console.print(f"  - {w}")
        raise typer.Exit(code=0)
    
    parsed = StructuredAnswer.model_validate(obj=obj)

    # Render answer with enriched citations
    console.print("\n[bold green]结构化回答（基于证据）[/bold green]")
    console.print(f"问题：{parsed.question}\n")

    _render_items("适用对象 / 范围", parsed.applicable_to, hits)
    _render_items("核心结论", parsed.key_conclusions, hits)
    _render_items("条件 / 资格 / 门槛", parsed.conditions, hits)
    _render_items("材料清单", parsed.materials, hits)
    _render_items("流程步骤", parsed.procedure, hits)
    _render_items("时间节点 / 截止日期", parsed.time_nodes, hits)
    _render_items("例外条款 / 坑点", parsed.exceptions_pitfalls, hits)
    _render_items("咨询渠道 / 官方入口", parsed.contact_channel, hits)

    if parsed.uncertainties:
        console.print("\n[bold yellow]不确定项（证据不足，需核对原文/补充信息）[/bold yellow]")
        for u in parsed.uncertainties:
            console.print(f"- {u}")

    if parsed.follow_up_questions:
        console.print("\n[bold]为了给出确定答案，建议你补充：[/bold]")
        for q in parsed.follow_up_questions:
            console.print(f"- {q}")

    if parsed.warnings:
        console.print("\n[bold]提醒：[/bold]")
        for w in parsed.warnings:
            console.print(f"- {w}")