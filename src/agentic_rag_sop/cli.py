"""
CLI Interface for Agentic RAG SOP & BPMN Knowledge.
Commands: ingest, query, chat, inspect-bpmn
Phase 5: Critic Verification Agent with self-correction and audit trail logging.
"""

import argparse
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode/Rich box drawing characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from agentic_rag_sop.config.settings import get_settings
from agentic_rag_sop.ingestion.pipeline import IngestionPipeline
from agentic_rag_sop.generation.chain import SOPRAGChain
from agentic_rag_sop.bpmn.parser import BPMNXMLParser

console = Console(safe_box=True, legacy_windows=False)


def run_ingest(args):
    """Execute ingestion pipeline on raw data directory."""
    settings = get_settings()
    target_path = args.path or settings.RAW_DATA_DIR
    console.print(f"[bold cyan]Starting ingestion from:[/bold cyan] {target_path}")
    console.print(f"[dim]Embedding Provider:[/dim] [green]{settings.EMBEDDING_PROVIDER}[/green] ({settings.SENTENCE_TRANSFORMER_MODEL if settings.EMBEDDING_PROVIDER == 'sentence-transformer' else settings.GEMINI_EMBEDDING_MODEL})")

    if getattr(args, "reset", False):
        console.print("[yellow]Reset flag detected: Clearing vector store collection...[/yellow]")

    pipeline = IngestionPipeline()
    try:
        results = pipeline.run(target_path, reset=getattr(args, "reset", False))
    except Exception as e:
        console.print(f"[bold red]Ingestion failed:[/bold red] {e}")
        sys.exit(1)

    if results.get("status") == "warning":
        console.print(f"[yellow]{results.get('message')}[/yellow]")
        return

    # Display summary table
    table = Table(title="Hasil Ingestion Dokumen", show_header=True, header_style="bold magenta")
    table.add_column("Metrik", style="cyan")
    table.add_column("Nilai", style="green")

    table.add_row("Sumber Data", results["source_path"])
    table.add_row("Jumlah File Diproses", str(results["files_processed"]))
    table.add_row("Potongan Chunk Diindeks", str(results["chunks_indexed"]))
    table.add_row("Total Vektor di ChromaDB", str(results["total_vectors_in_db"]))
    table.add_row("Tipe Dokumen", str(results.get("document_types", {})))
    table.add_row("Aktor Teridentifikasi", str(results.get("top_actors", {})))

    console.print(table)
    console.print("[bold green]Ingestion selesai dan tersimpan di database Chroma![/bold green]")


def run_query(args):
    """Run a single query against the RAG system with Phase 5 Critic Verification logging."""
    settings = get_settings()
    console.print(f"[dim]Model LLM:[/dim] [cyan]{settings.GEMINI_MODEL}[/cyan]")

    use_agent = not getattr(args, "no_rewrite", False)
    use_router = not getattr(args, "no_router", False)
    use_critic = not getattr(args, "no_critic", False)

    try:
        chain = SOPRAGChain()
        result = chain.invoke(
            question=args.question,
            k=args.k,
            document_type=args.type,
            actor=args.actor,
            process_name=args.process,
            use_query_agent=use_agent,
            use_router=use_router,
            use_critic=use_critic,
        )
    except Exception as e:
        console.print(f"[bold red]Error querying RAG system:[/bold red] {e}")
        sys.exit(1)

    # 1. Logging: Original Query
    console.print(f"[bold yellow]Original Query:[/bold yellow]\n{result['question']}\n")

    # 2. Logging: Rewritten Query
    if use_agent and result.get("rewritten_query"):
        console.print(f"[bold cyan]Rewritten Query:[/bold cyan]\n{result['rewritten_query']}\n")

    # 3. Logging: Router Decision
    route_decision = result.get("route_decision")
    if use_router and route_decision:
        console.print(
            f"[bold magenta]Router Decision:[/bold magenta]\n"
            f"Intent: [bold]{route_decision.intent}[/bold] | Knowledge Source: [bold]{route_decision.knowledge_source}[/bold]\n"
            f"Reason: {route_decision.reason}\n"
        )

    # 4. Logging: SOP Context
    sop_ctx = result.get("sop_context") or "Tidak ada konteks dokumen SOP yang dimuat."
    console.print(Panel(sop_ctx, title="[bold blue]SOP Context[/bold blue]", border_style="blue"))

    # 5. Logging: BPMN Context
    bpmn_ctx = result.get("bpmn_context") or "Tidak ada konteks graf proses BPMN yang dimuat."
    console.print(Panel(bpmn_ctx, title="[bold purple]BPMN Context[/bold purple]", border_style="purple"))

    # 6. Logging: Fused Context
    fused_text = result.get("fused_text")
    if not fused_text and result.get("fused_context"):
        fused_text = result["fused_context"].synthesized_context
    if fused_text:
        console.print(Panel(Markdown(fused_text), title="[bold cyan]Fused Context[/bold cyan]", border_style="cyan"))

    # 7. Logging: Selected Retriever
    selected_retriever = result.get("selected_retriever", "SOP Retriever")
    console.print(f"[bold green]Selected Retriever:[/bold green]\n{selected_retriever}\n")

    # 8. Logging: Critic Verification (Phase 5)
    critic_decision = result.get("critic_decision")
    if use_critic and critic_decision:
        was_revised = result.get("was_revised", False)
        if was_revised:
            status_text = "[bold yellow]REVISED (Auto-Corrected)[/bold yellow]"
        elif critic_decision.is_valid:
            status_text = "[bold green]PASSED (Valid & Grounded)[/bold green]"
        else:
            status_text = "[bold red]FAILED[/bold red]"

        critic_table = Table(show_header=True, header_style="bold yellow", box=None)
        critic_table.add_column("Pilar Audit", style="cyan", width=26)
        critic_table.add_column("Status", justify="center", width=12)
        critic_table.add_column("Catatan / Alasan", style="dim")

        def _status_badge(item):
            st = str(getattr(item, "status", item))
            if "PASSED" in st:
                return "[bold green]PASSED[/bold green]"
            elif "FAILED" in st:
                return "[bold red]FAILED[/bold red]"
            return "[dim]N/A[/dim]"

        critic_table.add_row("1. Factual Grounding", _status_badge(critic_decision.factual_grounding), critic_decision.factual_grounding.reason)
        critic_table.add_row("2. ICICoS Domain Integrity", _status_badge(critic_decision.domain_compliance), critic_decision.domain_compliance.reason)
        critic_table.add_row("3. BPMN Workflow Correctness", _status_badge(critic_decision.workflow_correctness), critic_decision.workflow_correctness.reason)
        critic_table.add_row("4. SOP Requirements Accuracy", _status_badge(critic_decision.sop_correctness), critic_decision.sop_correctness.reason)

        console.print(Panel(
            critic_table,
            title=f"[bold yellow]Critic Verification: {status_text} (Score: {critic_decision.confidence_score:.2f})[/bold yellow]",
            subtitle=f"[italic]{critic_decision.critique_summary}[/italic]",
            border_style="yellow",
        ))

    # Print Answer Panel
    title_suffix = " (Terverifikasi Critic)" if use_critic else ""
    console.print(Panel(Markdown(result["answer"]), title=f"[bold green]Jawaban Sistem{title_suffix}[/bold green]", border_style="green"))

    # Print Sources Table
    sources = result.get("sources", [])
    if sources:
        table = Table(title="Referensi Dokumen Terverifikasi", show_header=True, header_style="bold blue")
        table.add_column("No", justify="right", style="dim")
        table.add_column("Nama Dokumen", style="cyan")
        table.add_column("Tipe", style="magenta")
        table.add_column("Halaman", justify="center")
        table.add_column("Proses", style="yellow")
        table.add_column("Aktor Terkait", style="green")

        for idx, src in enumerate(sources, start=1):
            table.add_row(
                str(idx),
                src.get("document_name", "-"),
                src.get("document_type", "-").upper(),
                str(src.get("page_number", "-")),
                src.get("process_name", "-"),
                src.get("actor", "-"),
            )
        console.print(table)
    else:
        console.print("[dim yellow]Tidak ada sumber dokumen yang relevan.[/dim yellow]")


def run_chat(args):
    """Start interactive QA terminal chat session with Phase 5 Critic Verification."""
    settings = get_settings()
    console.print(Panel(
        f"[bold cyan]Asisten SOP & BPMN ICICoS (Phase 5 Critic Verification Agent)[/bold cyan]\n"
        f"LLM: [green]{settings.GEMINI_MODEL}[/green] | Embeddings: [green]{settings.EMBEDDING_PROVIDER}[/green]\n"
        f"Query Understanding: [green]{'Aktif' if not getattr(args, 'no_rewrite', False) else 'Nonaktif'}[/green] | "
        f"Router: [green]{'Aktif' if not getattr(args, 'no_router', False) else 'Nonaktif'}[/green] | "
        f"Context Fusion: [green]Aktif[/green] | "
        f"Critic Verification: [green]{'Aktif' if not getattr(args, 'no_critic', False) else 'Nonaktif'}[/green]\n"
        "Ketik [bold red]'exit'[/bold red] atau [bold red]'quit'[/bold red] untuk keluar.",
        title="Interactive Chat Mode"
    ))

    try:
        chain = SOPRAGChain()
    except Exception as e:
        console.print(f"[bold red]Gagal menginisialisasi RAG chain:[/bold red] {e}")
        sys.exit(1)

    use_agent = not getattr(args, "no_rewrite", False)
    use_router = not getattr(args, "no_router", False)
    use_critic = not getattr(args, "no_critic", False)

    while True:
        try:
            user_input = console.input("\n[bold yellow]User>[/bold yellow] ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "keluar"]:
                console.print("[cyan]Sampai jumpa![/cyan]")
                break

            console.print("[dim]Memproses query, fusi konteks, dan verifikasi Critic...[/dim]")
            result = chain.invoke(
                question=user_input,
                k=args.k,
                use_query_agent=use_agent,
                use_router=use_router,
                use_critic=use_critic,
            )

            if use_agent and result.get("rewritten_query"):
                console.print(f"[dim cyan]Rewritten Query:[/dim cyan] {result['rewritten_query']}")

            route_decision = result.get("route_decision")
            if use_router and route_decision:
                console.print(f"[dim magenta]Router Decision:[/dim magenta] {route_decision.intent} | Source: {route_decision.knowledge_source} (Reason: {route_decision.reason})")

            critic_decision = result.get("critic_decision")
            if use_critic and critic_decision:
                status_color = "green" if critic_decision.is_valid else "yellow"
                status_label = "REVISED" if result.get("was_revised") else ("PASSED" if critic_decision.is_valid else "FAILED")
                console.print(f"[{status_color}]Critic Audit: {status_label} (Score: {critic_decision.confidence_score:.2f})[/{status_color}]")

            console.print(Panel(Markdown(result["answer"]), title="[bold green]Asisten ICICoS[/bold green]", border_style="green"))

            sources = result.get("sources", [])
            if sources:
                src_summary = ", ".join(f"{s['document_name']} (Hal. {s['page_number']})" for s in sources)
                console.print(f"[dim]Sumber: {src_summary}[/dim]")

        except KeyboardInterrupt:
            console.print("\n[cyan]Sesi dihentikan.[/cyan]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")


def run_inspect_bpmn(args):
    """Parse and inspect a BPMN file structure."""
    path = Path(args.file)
    if not path.exists():
        console.print(f"[bold red]File not found:[/bold red] {path}")
        sys.exit(1)

    parser = BPMNXMLParser()
    process = parser.parse_file(path)

    console.print(f"[bold cyan]BPMN Process ID:[/bold cyan] {process.id}")
    console.print(f"[bold cyan]Process Name:[/bold cyan] {process.name}")
    console.print(f"[bold cyan]Lanes / Actors:[/bold cyan] {[lane.name for lane in process.lanes]}")
    console.print(f"[bold cyan]Total Activities (Tasks):[/bold cyan] {len(process.activities)}")
    console.print(f"[bold cyan]Total Gateways (Decisions):[/bold cyan] {len(process.gateways)}")
    console.print(f"[bold cyan]Total Sequence Flows:[/bold cyan] {len(process.sequence_flows)}")

    table = Table(title="Langkah Alur Proses (Process Steps)", show_header=True, header_style="bold magenta")
    table.add_column("Step ID", style="dim")
    table.add_column("Nama Langkah", style="cyan")
    table.add_column("Tipe", style="magenta")
    table.add_column("Aktor", style="green")
    table.add_column("Langkah Berikutnya", style="yellow")

    for step in process.steps:
        table.add_row(
            step.step_id,
            step.step_name,
            step.step_type,
            step.actor,
            "; ".join(step.succeeding_steps) if step.succeeding_steps else "End / Selesai",
        )

    console.print(table)


def main():
    """Main CLI parser."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG for SOP & BPMN Knowledge (Phase 5 Critic Verification Agent)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into Chroma vector store")
    ingest_parser.add_argument("--path", "-p", type=str, default=None, help="Directory or file path to ingest")
    ingest_parser.add_argument("--reset", "-r", action="store_true", help="Clear and reset the vector store before ingestion")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline with a question")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument("--k", type=int, default=None, help="Number of chunks to retrieve")
    query_parser.add_argument("--type", type=str, choices=["sop", "bpmn"], default=None, help="Filter by document type")
    query_parser.add_argument("--actor", type=str, default=None, help="Filter by actor role")
    query_parser.add_argument("--process", type=str, default=None, help="Filter by process name")
    query_parser.add_argument("--no-rewrite", action="store_true", help="Bypass Query Understanding Agent rewriting")
    query_parser.add_argument("--no-router", action="store_true", help="Bypass Router Agent classification")
    query_parser.add_argument("--no-critic", action="store_true", help="Bypass Critic Verification Agent audit")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive terminal chat session")
    chat_parser.add_argument("--k", type=int, default=None, help="Number of chunks to retrieve")
    chat_parser.add_argument("--no-rewrite", action="store_true", help="Bypass Query Understanding Agent rewriting")
    chat_parser.add_argument("--no-router", action="store_true", help="Bypass Router Agent classification")
    chat_parser.add_argument("--no-critic", action="store_true", help="Bypass Critic Verification Agent audit")

    # Inspect BPMN command
    bpmn_parser = subparsers.add_parser("inspect-bpmn", help="Inspect and display parsed BPMN graph topology")
    bpmn_parser.add_argument("file", type=str, help="Path to .bpmn or .xml file")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "chat":
        run_chat(args)
    elif args.command == "inspect-bpmn":
        run_inspect_bpmn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
