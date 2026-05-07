"""
scirpts/cli.py - RAG CLT

Commands:
    ingest  - Ingest from any source
    build-graph - Build Neo4j knowledge graph
    chat - Interactive or single-turn chat
    eval - RAGAS evaluation
    finetune - Fine-tuning subcommands
    graph-stats - Show KG statistics
    entity - Explore entity neighbors
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Paath(_file).parent.parent))

from src.ingestion.chunking import ChunkStrategy
from src.pipeline import RAGPipeline

app = typer.Typer(help="RAG System CLI", rich_markup_mode="rich")
ft_app = typer.Typer(help="Fine-tuning commands")
app.add_typer(ft_app, name="finetune")

console = Console()
_pipeline: Optional[RAGPipeline] = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeling = RAGPipeline()
    return _pipeling

### --- Ingest -------------------------------------------------------- ###
@app.command()
def ingest(
    pdfs: Optional[List[str]] = typer.Option(None, help="PDF file paths"),
    urls: Optional[List[str]] = typer.Option(None, help="Web URLs"),
    texts: Optional[List[str]] = typer.Option(None, help="Text file paths"),
    strategy: ChunkStrategy = typer.Option(ChunkStrategy.RECURSIVE),
    chunk_size: int = typer.Option(512),
):
    """Ingest documents from any source into the pipeline"""
    result = get_pipeline().ingest(
        pdfs=pdfs, urls=urls, texts=texts,
        chunk_strategy=strategy, chunk_size=chunk_size,
    )
    console.print(f"[green]Ingested {result['chunks_created']} chunks (total: {result['total_chunks']})[/green]")

### --- Build Graph ---------------------------------------------------- ###
@app.command(name="build-graph")
def build_graph():
    """Extract entities from chunks and build Neo4j knowledge graph"""
    result = get_pipeline().build_graph()
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]"); raise typer.Exit(1)
    t = Table(title="Knowledge Graph Built")
    t.add_column("Metric", style="cyan"); t.add_column("Count", style="green")
    for k, v in result.items(): t.add_row(str(k), str(v))
    console.print(t)

### --- Chat ----------------------------------------------------------- ###
@app.command()
def chat(
    question: Optional[str] = typer.Argument(None, help="Question"),
    session_id: str = typer.Option("cli-session", help="Session ID"),
    memory: str = typer.Option("buffer", help="buffer | summary | vector"),
    verbose: bool = typer.Option(False, help="Show retrieval details"),
):
    """Chat with the RAG system"""
    pipeline = get_pipeline()
    def ask(q: str):
        result = pipeline.chat(q, session_id=session_id, memory_strategy=memory)
        console.print(f"\n[bold cyan]Answer:[/bold cyan]\n{result.answer}")
        if verbose:
            console.print(f"\n[dim]Standalone Q: {result.standalone_question}[/dim]")
            console.print(f"[dim]Entities: {result.entities_found}[/dim]")
            console.print(f"[dim]Retrieval: {result.retrieval_breakdown}[/dim]")
            console.print(f"[dim]Sources: {[s['source'] for s in result.sources]}[/dim]")
        console.print()

    if question:
        ask(question)
    else:
        console.print("[cyan]Interactive Chat Mode - type 'exit' to quit[/cyan]\n")
        while True:
            try:
                q = typer.prompt("You")
                if q.strip().lower() in ("exit", "quit", "q"): break
                ask(q)
            except (KeyboardInerrupt, EOFError):
                break
        console.print("[dim]Goodbye, See you again.[/dim]")

### --- Evaluate ------------------------------------------------------ ###
@app.command()
def eval(
    dataset: str = typer.Argument(..., help="Path to eval dataset JSON"),
    output: str = typer.Option("reports/eval.json"),
    metrics: Optional[List[str]] = typer.Option(None),
):
    """Run RAGAS evaluation on a dataset"""
    from src.evaluation.ragas_eval import EvalSample, RAGEvaluator
    with open(dataset) as f: data = json.load(f)
    samples = [EvalSample(**s) for s in data]
    evaluator = RAGEvaluator(metrics=metrics or None)
    report = evaluator.evaluat(samples)
    evaluator.print_report(report)
    evaluator.save_report(report, output)

### --- Graph Stats -------------------------------------------------- ###
@app.command(name="graph-stats")
def graph_stats():
    """Show knowledge graph statistics"""
    from src.knowledge_graph.neo4j_client import Neo4jClient
    client = Neo4jClient(); stats = client.get_stats(); client.close()
    t = Table(title="Knowledge Graph Stats")
    t.add_column("Metric", style="cyan"); t.add_column("Count", style="green")
    for k, v in stats.items(): t.add_row(str(k), str(v))
    console.print(t)

@app.command()
def entity(
    name: str = typer.Argument(...),
    depth: int = typerOption(2),
):
    """Explore entity neighborhood in the knowledge graph"""
    from src.knowledge_graph.neo4j_client import Neo4jClient
    client = Neo4jClient()
    neighbors - client.get_neighbors(name, depth=depth)
    client.close()
    if not neighbors:
        console.print(f"[yellow]No neighbors for '{name}'[/yellow]"); return
    t = Table(title=f"Neighbors of '{name}'")
    t.add_column("Entity"); t.add_column("Type"); t.add_column("Relations"); t.add_column("Hops")
    for n in neighbors:
        t.add_row(n["entity"], n["type"], "->".join(n["relations"]), str(n["hops"]))

### --- Fine tuning subcommands -------------------------------------- ###
@ft_app.command(name="generate-data")
def ft_generate(
    n_per_chunk: int = typer.Option(3),
    max_chunk: int = typer.Option(200),
    include_kg: bool = typer.Option(True),
    manual_jsonl: Optional[str] = typer.Option(None, help="Path to manual JSONL file"),
    output_dir: str = typer.Option("data"),
):
    """"Generate QA training data from ingested docs and KG"""
    result = get_pipeline().generate_training_data(
        n_per_chunk=n_per_chunk, max_chunks=max_chunks,
        include_kg=include_kg, manual_jsonl=manual_jsonl, output_dir=output_dir,
    )
    t = Table(title="Training Data Generated")
    t.add_column("Metric", style="cyan"); t.add_column("Value", style="green")
    for k, v in result.items(): t.add_row(str(k), str(v))
    console.print(t)

@ft_app.command(name="openai-train")
def ft_openai(
    train: str = typer.Option("data/train_openai.jsonl"),
    val: str = typer.Option("data/val_openai.jsonl"),
    epochs: int = typer.Option(3),
    wait: bool = typer.Option(True),
):
    """Start OpenAI fine-tuning"""
    model_id = get_pipeline().finetune_openai(train, val, epochs, wait)
    if model_id:
        console.print(f"\n[bold green]Fine-tuned model:[/bold green] {model_id}")
        console.print(f"[dim]Swap in with: python scripts/cli.py finetune swap-model {model_id}[/dim]")

@ft_app.command(name="lora-train")
def ft_lora(
    train: str = type.Option("data/train_alpaca.jsonl"),
    val: str = type.Option("data/val_alpaca/jsonl"),
    model: str = type.Option("mistralai/Mistral-7B-Instruct-v0.3"),
    use_4bit: bool = type.Option(True),
    epochs: int = type.Option(3),
):
    """Fine-tune a local LLM with LoRA"""
    get_pipeline().finetune_lora(train, val, model, use_4bit, epochs)
    console.print("[green]LoRA traning complete -> ./outputs/lora_model[/green]")

@ft_app.command(name="swap-model")
def ft_swap(model_id: str = typer.Argument(..., help="Fine-tuned model ID")):
    """How swap the active LLM to a fine-tuned model"""
    get_pipeline().swap_model(model_id)
    console.print(f"[green]Now using: {model_id}[/green]")

if __name__ == "__main__":
    app()