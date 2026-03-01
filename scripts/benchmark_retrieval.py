#!/usr/bin/env python3
"""
ASCM Retrieval Benchmark Script.

Connects to the live AURA runtime and evaluates semantic retrieval performance
against a set of known architectural landmarks.

Usage:
    export OPENAI_API_KEY=...
    python3 scripts/benchmark_retrieval.py
"""
import sys
import os
import time
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_manager import config
from core.model_adapter import ModelAdapter
from memory.brain import Brain
from core.vector_store import VectorStore
from core.memory_types import RetrievalQuery

console = Console()

# Landmarks we expect to find in the AURA codebase
LANDMARKS = [
    {
        "query": "How are goals queued and managed?",
        "expected_terms": ["GoalQueue", "goal_queue.py"],
        "min_score": 0.75
    },
    {
        "query": "Where is the semantic memory vector store implemented?",
        "expected_terms": ["VectorStore", "vector_store.py"],
        "min_score": 0.75
    },
    {
        "query": "What agent handles codebase ingestion?",
        "expected_terms": ["IngestAgent", "ingest.py"],
        "min_score": 0.75
    },
    {
        "query": "logic for applying atomic file changes",
        "expected_terms": ["AtomicChangeSet", "file_tools.py"],
        "min_score": 0.70
    }
]

def main():
    console.print("[bold blue]ASCM Retrieval Benchmark[/bold blue]")
    
    # 1. Init Runtime
    try:
        project_root = Path(".")
        brain_db_path = project_root / config.get("brain_db_path", "memory/brain_v2.db")
        
        console.print(f"Connecting to Brain: [cyan]{brain_db_path}[/cyan]")
        brain = Brain(db_path=brain_db_path)
        
        adapter = ModelAdapter()
        # Ensure we can embed
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[red]Error: OPENAI_API_KEY not set. Cannot run benchmark.[/red]")
            sys.exit(1)
            
        vs = VectorStore(adapter, brain)
        stats = vs.stats()
        console.print(f"VectorStore Stats: [green]{stats}[/green]")
        
        if stats["record_count"] == 0:
            console.print("[yellow]Warning: VectorStore is empty. Run 'aura_cli' to trigger background sync first.[/yellow]")
            sys.exit(0)

    except Exception as e:
        console.print(f"[red]Setup Failed: {e}[/red]")
        sys.exit(1)

    # 2. Run Benchmarks
    table = Table(title="Retrieval Results")
    table.add_column("Query", style="cyan")
    table.add_column("Top Hit", style="green")
    table.add_column("Score", style="magenta")
    table.add_column("Found Expected?", style="bold")
    table.add_column("Latency (ms)", justify="right")

    total_hits = 0
    total_checks = 0
    
    for landmark in LANDMARKS:
        query_text = landmark["query"]
        expected = landmark["expected_terms"]
        
        start = time.time()
        query = RetrievalQuery(
            query_text=query_text,
            k=5,
            min_score=0.0 # Get everything to inspect ranking
        )
        hits = vs.search(query)
        elapsed_ms = (time.time() - start) * 1000
        
        if not hits:
            table.add_row(query_text, "NO HITS", "-", "[red]MISS[/red]", f"{elapsed_ms:.1f}")
            continue
            
        top_hit = hits[0]
        
        # Check if any expected term is in the top 3 hits
        found = False
        found_in = ""
        for i, hit in enumerate(hits[:3]):
            for term in expected:
                if term in hit.content or term in hit.source_ref:
                    found = True
                    found_in = f"(rank {i+1})"
                    break
            if found: break
        
        status = f"[green]PASS {found_in}[/green]" if found else f"[red]FAIL[/red] (Expected: {expected})"
        if found: total_hits += 1
        total_checks += 1
        
        # Highlight source
        source_display = f"{top_hit.source_ref}\n{top_hit.content[:60]}..."
        
        table.add_row(
            query_text,
            source_display,
            f"{top_hit.score:.3f}",
            status,
            f"{elapsed_ms:.1f}"
        )

    console.print(table)
    
    # Summary
    score = (total_hits / total_checks) * 100 if total_checks > 0 else 0
    console.print(f"\nOverall Recall Score: [bold]{score:.1f}%[/bold]")
    
    if score < 50:
        console.print("[red]FAILURE: Retrieval quality is below 50%. Check embeddings or chunking logic.[/red]")
        sys.exit(1)
    else:
        console.print("[green]SUCCESS: Retrieval system is operational.[/green]")

if __name__ == "__main__":
    main()
