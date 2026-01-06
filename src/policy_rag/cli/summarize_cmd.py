from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from policy_rag.config.settings import Settings
from policy_rag.index.chroma_store import ChromaStore
from policy_rag.llm.llm_client import OllamaClient, ChatMessage
from policy_rag.prompts.policy_card_prompt import SYSTEM_PROMPT, USER_TEMPLATE
from policy_rag.schemas.structured_answer import StructuredAnswer
from policy_rag.utils.json_extract import extract_first_json
from policy_rag.ingestion.indexing import load_docs_meta
from policy_rag.retrieval.quote_verify import quote_in_text

console = Console()

def _pick_representative_sources(documents, metadatas, max_sources: int = 16):
    """
    v0 策略：按 page_number 去重，每页取一个“文本更长”的 chunk
    """
    by_page = {}
    for doc, md in zip(documents, metadatas):
        if not doc:
            continue
        page = md.get("page_number")
        if page is None:
            continue
        try:
            page = int(page)
        except Exception:
            continue

        text = str(doc).strip()
        if not text:
            continue

        cur = by_page.get(page)
        if cur is None or len(text) > len(cur["text"]):
            by_page[page] = {"text": text, "md": md}

    pages = sorted(by_page.keys())
    picked = []
    for p in pages:
        picked.append(by_page[p])
        if len(picked) >= max_sources:
            break

    if not picked:
        for doc, md in zip(documents, metadatas):
            t = str(doc or "").strip()
            if t:
                picked.append({"text": t, "md": md})
            if len(picked) >= max_sources:
                break

    return picked

def _format_sources(picked, max_chars_per_source: int = 900) -> str:
    blocks = []
    for i, it in enumerate(picked, start=1):
        md = it["md"] or {}
        did = str(md.get("doc_id", "") or "")
        title = str(md.get("title", "") or "")
        page = md.get("page_number", "")
        sec = str(md.get("section_path", "") or "")

        text = it["text"]
        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source] + "..."

        header = f"[{i}] doc_id={did} title={title} page={page} section={sec}".strip()
        blocks.append(header + "\n" + text)
    return "\n\n--\n\n".join(blocks)

def _render_items(title: str, items, picked):
    if not items:
        return
    console.print(f"\n[bold]{title}[/bold]")
    for i, it in enumerate(items, start=1):
        console.print(f"{i}. {it.text}  [dim]({it.confidence})[/dim]")

        for cit in it.citations:
            sid = cit.source_id
            if sid < 1 or sid > len(picked):
                console.print(f"   - [red]Citation error[/red]: source_id={sid} out of range")
                continue
            md = picked[sid - 1]["md"] or {}
            title0 = str(md.get("title", "") or "")
            did = str(md.get("doc_id", "") or "")
            page = md.get("page_number", "")
            quote = cit.quote.strip().replace("\n", " ")
            if len(quote) > 120:
                quote = quote[:120].rstrip() + "…"
            source_text = picked[sid - 1]["text"] or ""
            ok = quote_in_text(quote, source_text)
            tag = "[green]QUOTE_OK[/green]" if ok else "[bold red]QUOTE_MISSING[/bold red]"

            console.print(f"   - 引用：{title0 or did} | doc_id={did} | p.{page} | “{quote}”  {tag}")
            if not ok:
                console.print("     [red]提示[/red]：quote 未在该 source chunk 中命中，可能是模型改写/拼接/省略号导致。")


def summarize(
    doc_id: str = ...,
    max_sources: int = 16
):
    settings = Settings.from_repo_root()

    docs_meta = load_docs_meta(settings.docs_csv_path)
    meta = docs_meta.get(doc_id)
    if meta is None:
        console.print(f"[bold red]ERROR[/bold red] doc_id not found in docs.csv: {doc_id}")
        raise typer.Exit(code=1)
    
    store = ChromaStore(
        persist_dir=settings.index_dir / "chroma",
        collection_name=settings.chroma_collection,
    )

    if store.count() == 0:
        console.print("[bold red]ERROR[/bold red] Chroma collection is empty. Run index-chunks first.")
        raise typer.Exit(code=1)
    
    got = store.get(where={"doc_id": doc_id}, limit=5000)
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []

    if not docs:
        console.print(f"[bold red]ERROR[/bold red] No chunks found for doc_id={doc_id}. Did you index-chunks?")
        raise typer.Exit(code=1)
    
    picked = _pick_representative_sources(docs, metas, max_sources=max_sources)

    pages = []
    for it in picked:
        pg = it["md"].get("page_number")
        try:
            pages.append(int(pg))
        except Exception:
            pass

    pages = sorted(set(pages))

    console.print(f"\n[bold]Summarize[/bold] doc_id={doc_id}")
    console.print(f"  title: {meta.title}")
    console.print(f"  picked_sources: {len(picked)}")
    if pages:
        console.print(f"  covered_pages: {pages[:20]}{'...' if len(pages) > 20 else ''}")

    sources_str = _format_sources(picked=picked, max_chars_per_source=900)

    user_prompt = USER_TEMPLATE.format(
        doc_id=meta.doc_id,
        title=meta.title,
        category=meta.category,
        publish_date=meta.publish_date,
        effective_date=meta.effective_date,
        status=meta.status,
        sources=sources_str,
    )

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
    parsed = StructuredAnswer.model_validate(obj)

    console.print("\n[bold green]政策速览卡片（基于证据）[/bold green]")
    console.print(f"制度：{parsed.question}")

    _render_items("适用对象 / 范围", parsed.applicable_to, picked)
    _render_items("核心结论", parsed.key_conclusions, picked)
    _render_items("条件 / 资格 / 门槛", parsed.conditions, picked)
    _render_items("材料清单", parsed.materials, picked)
    _render_items("流程步骤", parsed.procedure, picked)
    _render_items("时间节点 / 截止日期", parsed.time_nodes, picked)
    _render_items("例外条款 / 坑点", parsed.exceptions_pitfalls, picked)
    _render_items("咨询渠道 / 官方入口", parsed.contact_channel, picked)

    if parsed.uncertainties:
        console.print("\n[bold yellow]不确定项（证据不足，需核对原文/补充信息）[/bold yellow]")
        for u in parsed.uncertainties:
            console.print(f"- {u}")

    if parsed.warnings:
        console.print("\n[bold]提醒：[/bold]")
        for w in parsed.warnings:
            console.print(f"- {w}")