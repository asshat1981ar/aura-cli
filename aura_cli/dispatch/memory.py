"""Dispatch handlers for memory-related commands (B4)."""
from __future__ import annotations

import sys

from aura_cli.dispatch._helpers import _print_json_payload


def handle_memory_search(ctx) -> int:
    from core.memory_types import RetrievalQuery

    vector_store = ctx.runtime["vector_store"]
    query = RetrievalQuery(
        query_text=ctx.args.query,
        k=ctx.args.limit
    )
    hits = vector_store.search(query)

    if getattr(ctx.args, "json", False):
        payload = {
            "query": ctx.args.query,
            "hits": [
                {
                    "score": hit.score,
                    "source_ref": hit.source_ref,
                    "content_preview": hit.content[:200] + "..." if len(hit.content) > 200 else hit.content
                }
                for hit in hits
            ]
        }
        _print_json_payload(payload, parsed=ctx.parsed, indent=2)
        return 0

    if not hits:
        print(f"No results found for '{ctx.args.query}'")
        return 0

    print(f"Memory Search Results for '{ctx.args.query}':\n")
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] Score: {hit.score:.3f} | Source: {hit.source_ref}")
        print(f"Content: {hit.content[:200]}...")
        print("-" * 40)
    return 0


def handle_memory_reindex(ctx) -> int:
    from core.project_syncer import ProjectKnowledgeSyncer

    runtime = ctx.runtime
    vector_store = runtime["vector_store"]
    model_adapter = runtime["model_adapter"]

    rebuild_stats = vector_store.rebuild({
        "exclude_source_types": ["file"],
        "drop_existing_embeddings": True,
    })
    syncer = ProjectKnowledgeSyncer(vector_store, None, project_root=str(ctx.project_root))
    sync_stats = syncer.sync_all(force=True)

    payload = {
        "status": "ok" if "error" not in rebuild_stats else "error",
        "embedding_model": model_adapter.model_id(),
        "embedding_dims": model_adapter.dimensions(),
        "rebuild": rebuild_stats,
        "project_sync": sync_stats,
    }

    if getattr(ctx.args, "json", False):
        _print_json_payload(payload, parsed=ctx.parsed, indent=2)
        return 0 if payload["status"] == "ok" else 1

    print("Semantic memory reindex complete.")
    print(f"Embedding model: {payload['embedding_model']} ({payload['embedding_dims']} dims)")
    print(
        "Non-file records rebuilt: "
        f"{rebuild_stats.get('embeddings_written', 0)}/{rebuild_stats.get('records_seen', 0)}"
    )
    print(
        "Project sync: "
        f"{sync_stats.get('files_processed', 0)} files processed, "
        f"{sync_stats.get('chunks_created', 0)} chunks created, "
        f"{sync_stats.get('files_skipped', 0)} skipped"
    )
    if payload["status"] != "ok":
        print(f"Error: {rebuild_stats.get('error')}", file=sys.stderr)
        return 1
    return 0
