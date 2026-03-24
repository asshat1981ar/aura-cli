"""
Weaviate MCP Server — vector database operations for semantic search and embeddings.

Tools exposed:
  Vectors : vector_upsert, vector_search, hybrid_search
  Schema  : schema_create
  Import  : batch_import

Endpoints:
  GET  /tools          → list all tools as MCP descriptors
  POST /call           → invoke a tool by name with args dict
  GET  /tool/{name}    → descriptor for a single tool
  GET  /health         → health check
  GET  /metrics        → uptime and per-tool call/error counts

Start:
  uvicorn tools.weaviate_mcp:app --port 8017

Auth (optional):
  Set WEAVIATE_MCP_TOKEN env var
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from fastapi import Depends, FastAPI, Header, HTTPException
from tools.mcp_types import ToolCallRequest, ToolResult
from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Lazy-loaded Weaviate client
# ---------------------------------------------------------------------------

_weaviate_client = None
_weaviate_available = True

try:
    import weaviate as _weaviate_mod
except ImportError:
    _weaviate_mod = None  # type: ignore[assignment]
    _weaviate_available = False


def _get_client():
    """Lazy-load and return the Weaviate client singleton."""
    global _weaviate_client
    if not _weaviate_available:
        raise RuntimeError(
            "weaviate-client Python package is not installed. "
            "Install it with: pip install weaviate-client"
        )
    if _weaviate_client is None:
        url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        _weaviate_client = _weaviate_mod.Client(url=url)
    return _weaviate_client


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Weaviate MCP", version="1.0.0")
_TOKEN = os.getenv("WEAVIATE_MCP_TOKEN", "")
_SERVER_START = time.time()
_call_counts: Dict[str, int] = {}
_call_errors: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not _TOKEN:
        return
    if authorization != f"Bearer {_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Tool descriptors (MCP schema format)
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: Dict[str, Dict] = {
    "vector_upsert": {
        "description": "Upsert a vector object into a Weaviate collection.",
        "input": {
            "collection": {"type": "string", "description": "Collection (class) name", "required": True},
            "id": {"type": "string", "description": "Object UUID", "required": True},
            "properties": {"type": "object", "description": "Object properties", "required": True},
            "vector": {"type": "array", "description": "Optional embedding vector (list of floats)"},
        },
    },
    "vector_search": {
        "description": "Search a Weaviate collection by vector similarity.",
        "input": {
            "collection": {"type": "string", "description": "Collection (class) name", "required": True},
            "query_vector": {"type": "array", "description": "Query embedding vector", "required": True},
            "limit": {"type": "integer", "description": "Max results to return", "default": 10},
        },
    },
    "schema_create": {
        "description": "Create a collection schema in Weaviate.",
        "input": {
            "collection": {"type": "string", "description": "Collection (class) name", "required": True},
            "properties": {"type": "array", "description": "List of property definitions", "required": True},
        },
    },
    "batch_import": {
        "description": "Batch import objects into a Weaviate collection.",
        "input": {
            "collection": {"type": "string", "description": "Collection (class) name", "required": True},
            "objects": {"type": "array", "description": "List of object dicts with 'properties' and optional 'id', 'vector'", "required": True},
        },
    },
    "hybrid_search": {
        "description": "Hybrid keyword + vector search in a Weaviate collection.",
        "input": {
            "collection": {"type": "string", "description": "Collection (class) name", "required": True},
            "query": {"type": "string", "description": "Search query text", "required": True},
            "limit": {"type": "integer", "description": "Max results to return", "default": 10},
            "alpha": {"type": "number", "description": "Balance between keyword (0) and vector (1) search", "default": 0.5},
        },
    },
}


def _build_descriptor(name: str) -> Dict:
    schema = _TOOL_SCHEMAS[name]
    return {
        "name": name,
        "description": schema["description"],
        "inputSchema": {
            "type": "object",
            "properties": schema.get("input", {}),
            "required": [k for k, v in schema.get("input", {}).items() if v.get("required")],
        },
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _vector_upsert(args: Dict) -> Any:
    collection = args.get("collection", "").strip()
    if not collection:
        raise ValueError("'collection' is required.")
    obj_id = args.get("id", "").strip()
    if not obj_id:
        raise ValueError("'id' is required.")
    properties = args.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("'properties' is required and must be a dict.")
    vector = args.get("vector")

    client = _get_client()
    data_object = {
        "class": collection,
        "id": obj_id,
        "properties": properties,
    }
    if vector is not None:
        if not isinstance(vector, list):
            raise ValueError("'vector' must be a list of floats.")
        data_object["vector"] = vector

    # Use batch for upsert semantics
    try:
        client.data_object.create(
            data_object=properties,
            class_name=collection,
            uuid=obj_id,
            vector=vector,
        )
        action = "created"
    except Exception:
        # If object exists, update it
        client.data_object.update(
            data_object=properties,
            class_name=collection,
            uuid=obj_id,
            vector=vector,
        )
        action = "updated"

    return {"collection": collection, "id": obj_id, "action": action}


def _vector_search(args: Dict) -> Any:
    collection = args.get("collection", "").strip()
    if not collection:
        raise ValueError("'collection' is required.")
    query_vector = args.get("query_vector")
    if not isinstance(query_vector, list):
        raise ValueError("'query_vector' is required and must be a list of floats.")
    limit = int(args.get("limit", 10))

    client = _get_client()
    result = (
        client.query
        .get(collection)
        .with_near_vector({"vector": query_vector})
        .with_limit(limit)
        .with_additional(["id", "distance"])
        .do()
    )

    data = result.get("data", {}).get("Get", {}).get(collection, [])
    return {"collection": collection, "results": data, "count": len(data)}


def _schema_create(args: Dict) -> Any:
    collection = args.get("collection", "").strip()
    if not collection:
        raise ValueError("'collection' is required.")
    properties = args.get("properties")
    if not isinstance(properties, list):
        raise ValueError("'properties' is required and must be a list of dicts.")

    client = _get_client()
    class_obj = {
        "class": collection,
        "properties": properties,
    }
    client.schema.create_class(class_obj)
    return {"collection": collection, "created": True, "property_count": len(properties)}


def _batch_import(args: Dict) -> Any:
    collection = args.get("collection", "").strip()
    if not collection:
        raise ValueError("'collection' is required.")
    objects = args.get("objects")
    if not isinstance(objects, list):
        raise ValueError("'objects' is required and must be a list of dicts.")

    client = _get_client()
    imported = 0
    errors_list: List[str] = []

    with client.batch as batch:
        for obj in objects:
            if not isinstance(obj, dict):
                errors_list.append(f"Skipped non-dict object: {obj}")
                continue
            props = obj.get("properties", {})
            obj_id = obj.get("id")
            vector = obj.get("vector")
            try:
                batch.add_data_object(
                    data_object=props,
                    class_name=collection,
                    uuid=obj_id,
                    vector=vector,
                )
                imported += 1
            except Exception as exc:
                errors_list.append(str(exc))

    return {
        "collection": collection,
        "imported": imported,
        "errors": errors_list,
        "error_count": len(errors_list),
    }


def _hybrid_search(args: Dict) -> Any:
    collection = args.get("collection", "").strip()
    if not collection:
        raise ValueError("'collection' is required.")
    query = args.get("query", "").strip()
    if not query:
        raise ValueError("'query' is required.")
    limit = int(args.get("limit", 10))
    alpha = float(args.get("alpha", 0.5))

    client = _get_client()
    result = (
        client.query
        .get(collection)
        .with_hybrid(query=query, alpha=alpha)
        .with_limit(limit)
        .with_additional(["id", "score"])
        .do()
    )

    data = result.get("data", {}).get("Get", {}).get(collection, [])
    return {"collection": collection, "query": query, "alpha": alpha, "results": data, "count": len(data)}


# Map tool names → handler functions
_TOOL_HANDLERS = {
    "vector_upsert": _vector_upsert,
    "vector_search": _vector_search,
    "schema_create": _schema_create,
    "batch_import": _batch_import,
    "hybrid_search": _hybrid_search,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(_: None = Depends(_check_auth)):
    return {
        "status": "ok",
        "tool_count": len(_TOOL_HANDLERS),
        "server": "weaviate_mcp",
        "version": "1.0.0",
        "weaviate_available": _weaviate_available,
    }


@app.get("/tools")
async def list_tools(_: None = Depends(_check_auth)) -> List[Dict]:
    return [_build_descriptor(name) for name in _TOOL_SCHEMAS]


@app.get("/tool/{name}")
async def get_tool(name: str, _: None = Depends(_check_auth)) -> Dict:
    if name not in _TOOL_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")
    return _build_descriptor(name)


@app.post("/call")
async def call_tool(request: ToolCallRequest, _: None = Depends(_check_auth)) -> ToolResult:
    name = request.tool_name
    handler = _TOOL_HANDLERS.get(name)
    if not handler:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")

    _call_counts[name] = _call_counts.get(name, 0) + 1
    t0 = time.time()
    try:
        result = handler(request.args)
        elapsed = round((time.time() - t0) * 1000, 2)
        log_json("INFO", "weaviate_mcp_tool_called", details={"tool": name, "elapsed_ms": elapsed})
        return ToolResult(tool_name=name, result=result, elapsed_ms=elapsed)
    except ValueError as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("WARN", "weaviate_mcp_tool_bad_args", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=str(exc), elapsed_ms=elapsed)
    except Exception as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("ERROR", "weaviate_mcp_tool_error", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=f"Internal error: {exc}", elapsed_ms=elapsed)


@app.get("/metrics")
async def get_metrics(_: None = Depends(_check_auth)) -> Dict:
    uptime_s = round(time.time() - _SERVER_START, 1)
    total_calls = sum(_call_counts.values())
    total_errors = sum(_call_errors.values())
    per_tool = {
        name: {
            "calls": _call_counts.get(name, 0),
            "errors": _call_errors.get(name, 0),
        }
        for name in _TOOL_SCHEMAS
    }
    return {
        "uptime_seconds": uptime_s,
        "total_calls": total_calls,
        "total_errors": total_errors,
        "error_rate": round(total_errors / max(total_calls, 1), 4),
        "tools": per_tool,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from core.config_manager import config as _cfg
    port = int(os.getenv("WEAVIATE_MCP_PORT", _cfg.get_mcp_server_port("weaviate", default=8017)))
    uvicorn.run("tools.weaviate_mcp:app", host="0.0.0.0", port=port, reload=False)
