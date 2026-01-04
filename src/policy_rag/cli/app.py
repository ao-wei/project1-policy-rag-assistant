# 将项目变成一个“可执行的命令行工具“，并组织各个子命令
from __future__ import annotations

# 一个用类型注解来快速写 CLI 的框架
import typer

# 导入用于美化终端输出的库
from rich.console import Console
from rich.table import Table

from policy_rag.config.settings import Settings
from policy_rag.ingestion.validators import validate_docs_csv
from policy_rag.cli.parse_cmd import parse_pdf
from policy_rag.cli.chunk_cmd import chunk_pages
from policy_rag.cli.index_cmd import index_chunks
from policy_rag.cli.search_cmd import search
from policy_rag.cli.ask_cmd import ask
from policy_rag.cli.summarize_cmd import summarize

# 创建一个 CLI“应用对象“，后续所有命令都挂在它下面，关闭自动补全
# app是一个 Typer 对象，这个对象实现了__call__（可调用协议），可以像函数一样被调用
app = typer.Typer(add_completion=False)

# 创建 Rich 控制台对象
console = Console()

@app.command("validate-metadata")
def validate_metadata():
    """
    Validate data/metadata/docs.csv:
    - required columns exist
    - required fields non-empty
    - doc_id unique
    - file_path exists
    - date format is YYYY-MM-DD
    """
    settings = Settings.from_repo_root()

    issues = validate_docs_csv(settings.docs_csv_path, settings.repo_root)

    errors = [i for i in issues if i.level == "ERROR"]
    warns = [i for i in issues if i.level == "WARN"]

    if issues:
        table = Table(title="Metadata Validation Report", show_lines=True)
        table.add_column("Level", style="bold")
        table.add_column("Row")
        table.add_column("Field")
        table.add_column("Message")

        for it in issues:
            table.add_row(
                it.level,
                "" if it.row is None else str(it.row),
                "" if it.field is None else it.field,
                it.message,
            )

        console.print(table)

    if errors:
        console.print(f"[bold red]FAIL[/bold red] ({len(errors)} errors, {len(warns)} warnings)")
        raise typer.Exit(code=1)
        
    console.print(f"[bold green]PASS[/bold green] (0 errors, {len(warns)} warnings)")

@app.command("parse-pdf")
def parse_pdf_cmd(doc_id: str | None = typer.Option(None, help="Parse a single doc_id from docs.csv"),
                  all_docs: bool = typer.Option(False, help="Parse all docs in docs.csv")
                  ):
    parse_pdf(doc_id=doc_id, all_docs=all_docs)

@app.command("chunk-pages")
def chunk_pages_cmd(
    doc_id: str = typer.Option(..., help="Target doc_id"),
    chunk_size: int = typer.Option(1000, help="Chunk size in characters"),
    overlap: int = typer.Option(150, help="Overlap in characters"),
    min_chunk_chars: int = typer.Option(80, help="Drop too-short chunks"),
):
    chunk_pages(doc_id, chunk_size, overlap, min_chunk_chars)

@app.command("index-chunks")
def index_chunks_cmd(
    doc_id: str = typer.Option(..., help="Target doc_id"),
    batch_size: int = typer.Option(32, help="Embedding batch size"),
):
    index_chunks(doc_id, batch_size)

@app.command("search")
def search_cmd(
    query: str = typer.Option(..., help="User question/query text"),
    top_k: int = typer.Option(8, help="Number of chunks to retrieve"),
    doc_id: str | None = typer.Option(None, help="Restrict to doc_id"),
    category: str | None = typer.Option(None, help="Restrict to category"),
    show_full: bool = typer.Option(False, help="Show full chunk text"),
    use_gate: bool = typer.Option(True, help="Enable evidence gate (recommended)")
):
    search(
        query,
        top_k,
        doc_id,
        category,
        show_full,
        use_gate,
    )

@app.command("ask")
def ask_cmd(
    query: str = typer.Option(..., help="User question"),
    top_k: int = typer.Option(8, help="Top-k retrieval"),
    doc_id: str | None = typer.Option(None, help="Restrict to doc_id"),
    category: str | None = typer.Option(None, help="Restrict to category"),
    use_gate: bool = typer.Option(True, help="Enable evidence gate"),
    show_evidence: bool = typer.Option(True, help="Print evidence table"),
):
    ask(query=query, top_k=top_k, doc_id=doc_id, category=category, use_gate=use_gate, show_evidence=show_evidence)

@app.command("summarize")
def summarize_cmd(
    doc_id: str = typer.Option(..., help="Target doc_id"),
    max_sources: int = typer.Option(16, help="Max evidence chunks"),
):
    summarize(doc_id=doc_id, max_sources=max_sources)

def main():
    app()

if __name__ == "__main__":
    main()