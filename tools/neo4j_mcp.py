"""
Neo4j MCP Server — graph database operations for code knowledge graphs.

Tools exposed:
  Queries       : query_cypher, find_paths, schema_info
  Mutations     : create_node, create_relationship
  Import        : import_codebase_graph

Endpoints:
  GET  /tools          → list all tools as MCP descriptors
  POST /call           → invoke a tool by name with args dict
  GET  /tool/{name}    → descriptor for a single tool
  GET  /health         → health check
  GET  /metrics        → uptime and per-tool call/error counts

Start:
  uvicorn tools.neo4j_mcp:app --port 8013

Auth (optional):
  Set NEO4J_MCP_TOKEN env var
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
# Lazy-loaded neo4j driver
# ---------------------------------------------------------------------------

_neo4j_driver = None
_neo4j_available = True

try:
    import neo4j as _neo4j_mod
except ImportError:
    _neo4j_mod = None  # type: ignore[assignment]
    _neo4j_available = False


def _get_driver():
    """Lazy-load and return the Neo4j driver singleton."""
    global _neo4j_driver
    if not _neo4j_available:
        raise RuntimeError(
            "neo4j Python driver is not installed. "
            "Install it with: pip install neo4j"
        )
    if _neo4j_driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        _neo4j_driver = _neo4j_mod.GraphDatabase.driver(uri, auth=(user, password))
    return _neo4j_driver


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Neo4j MCP", version="1.0.0")
_TOKEN = os.getenv("NEO4J_MCP_TOKEN", "")
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
    "query_cypher": {
        "description": "Execute a read-only Cypher query against the Neo4j database.",
        "input": {
            "query": {"type": "string", "description": "Cypher query string", "required": True},
            "params": {"type": "object", "description": "Query parameters (optional)"},
        },
    },
    "create_node": {
        "description": "Create a node with a label and properties.",
        "input": {
            "label": {"type": "string", "description": "Node label", "required": True},
            "properties": {"type": "object", "description": "Node properties", "required": True},
        },
    },
    "create_relationship": {
        "description": "Create a relationship between two nodes.",
        "input": {
            "from_label": {"type": "string", "description": "Source node label", "required": True},
            "from_key": {"type": "string", "description": "Source node match property name", "required": True},
            "from_value": {"type": "string", "description": "Source node match property value", "required": True},
            "to_label": {"type": "string", "description": "Target node label", "required": True},
            "to_key": {"type": "string", "description": "Target node match property name", "required": True},
            "to_value": {"type": "string", "description": "Target node match property value", "required": True},
            "rel_type": {"type": "string", "description": "Relationship type", "required": True},
        },
    },
    "find_paths": {
        "description": "Find shortest paths between two nodes.",
        "input": {
            "from_label": {"type": "string", "description": "Source node label", "required": True},
            "from_key": {"type": "string", "description": "Source node match property name", "required": True},
            "from_value": {"type": "string", "description": "Source node match property value", "required": True},
            "to_label": {"type": "string", "description": "Target node label", "required": True},
            "to_key": {"type": "string", "description": "Target node match property name", "required": True},
            "to_value": {"type": "string", "description": "Target node match property value", "required": True},
            "max_depth": {"type": "integer", "description": "Maximum path depth", "default": 5},
        },
    },
    "schema_info": {
        "description": "Return node labels, relationship types, and constraint info from the database.",
        "input": {},
    },
    "import_codebase_graph": {
        "description": "Import a codebase structure into the graph as nodes and relationships.",
        "input": {
            "nodes": {"type": "array", "description": "List of node dicts with 'label' and 'properties'", "required": True},
            "relationships": {"type": "array", "description": "List of relationship dicts", "required": True},
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

def _query_cypher(args: Dict) -> Any:
    query = args.get("query", "").strip()
    if not query:
        raise ValueError("'query' is required.")
    params = args.get("params", {}) or {}
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(query, **params)
        records = [dict(record) for record in result]
    return {"records": records, "count": len(records)}


def _create_node(args: Dict) -> Any:
    label = args.get("label", "").strip()
    if not label:
        raise ValueError("'label' is required.")
    properties = args.get("properties")
    if not properties or not isinstance(properties, dict):
        raise ValueError("'properties' is required and must be a dict.")
    driver = _get_driver()
    props_str = ", ".join(f"{k}: ${k}" for k in properties)
    cypher = f"CREATE (n:{label} {{{props_str}}}) RETURN n"
    with driver.session() as session:
        result = session.run(cypher, **properties)
        record = result.single()
    return {"created": dict(record["n"]) if record else None, "label": label}


def _create_relationship(args: Dict) -> Any:
    required = ["from_label", "from_key", "from_value", "to_label", "to_key", "to_value", "rel_type"]
    for field in required:
        if not args.get(field, "").strip():
            raise ValueError(f"'{field}' is required.")
    driver = _get_driver()
    cypher = (
        f"MATCH (a:{args['from_label']} {{{args['from_key']}: $from_value}}), "
        f"(b:{args['to_label']} {{{args['to_key']}: $to_value}}) "
        f"CREATE (a)-[r:{args['rel_type']}]->(b) RETURN type(r) AS rel_type"
    )
    with driver.session() as session:
        result = session.run(cypher, from_value=args["from_value"], to_value=args["to_value"])
        record = result.single()
    return {"relationship": record["rel_type"] if record else None}


def _find_paths(args: Dict) -> Any:
    required = ["from_label", "from_key", "from_value", "to_label", "to_key", "to_value"]
    for field in required:
        if not args.get(field, "").strip():
            raise ValueError(f"'{field}' is required.")
    max_depth = int(args.get("max_depth", 5))
    driver = _get_driver()
    cypher = (
        f"MATCH (a:{args['from_label']} {{{args['from_key']}: $from_value}}), "
        f"(b:{args['to_label']} {{{args['to_key']}: $to_value}}), "
        f"p = shortestPath((a)-[*..{max_depth}]-(b)) "
        f"RETURN p"
    )
    with driver.session() as session:
        result = session.run(cypher, from_value=args["from_value"], to_value=args["to_value"])
        paths = []
        for record in result:
            path = record["p"]
            paths.append({
                "nodes": [dict(node) for node in path.nodes],
                "length": len(path),
            })
    return {"paths": paths, "count": len(paths)}


def _schema_info(args: Dict) -> Any:
    driver = _get_driver()
    with driver.session() as session:
        labels_result = session.run("CALL db.labels()")
        labels = [r["label"] for r in labels_result]

        rel_result = session.run("CALL db.relationshipTypes()")
        rel_types = [r["relationshipType"] for r in rel_result]

        constraint_result = session.run("SHOW CONSTRAINTS")
        constraints = [dict(r) for r in constraint_result]

    return {"labels": labels, "relationship_types": rel_types, "constraints": constraints}


def _import_codebase_graph(args: Dict) -> Any:
    nodes = args.get("nodes")
    relationships = args.get("relationships")
    if not isinstance(nodes, list):
        raise ValueError("'nodes' is required and must be a list.")
    if not isinstance(relationships, list):
        raise ValueError("'relationships' is required and must be a list.")

    driver = _get_driver()
    created_nodes = 0
    created_rels = 0

    with driver.session() as session:
        for node in nodes:
            label = node.get("label", "Node")
            props = node.get("properties", {})
            props_str = ", ".join(f"{k}: ${k}" for k in props)
            cypher = f"MERGE (n:{label} {{{props_str}}}) RETURN n"
            session.run(cypher, **props)
            created_nodes += 1

        for rel in relationships:
            from_label = rel.get("from_label", "Node")
            from_key = rel.get("from_key", "name")
            from_value = rel.get("from_value", "")
            to_label = rel.get("to_label", "Node")
            to_key = rel.get("to_key", "name")
            to_value = rel.get("to_value", "")
            rel_type = rel.get("rel_type", "RELATES_TO")
            cypher = (
                f"MATCH (a:{from_label} {{{from_key}: $from_value}}), "
                f"(b:{to_label} {{{to_key}: $to_value}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) RETURN type(r)"
            )
            session.run(cypher, from_value=from_value, to_value=to_value)
            created_rels += 1

    return {"created_nodes": created_nodes, "created_relationships": created_rels}


# Map tool names → handler functions
_TOOL_HANDLERS = {
    "query_cypher": _query_cypher,
    "create_node": _create_node,
    "create_relationship": _create_relationship,
    "find_paths": _find_paths,
    "schema_info": _schema_info,
    "import_codebase_graph": _import_codebase_graph,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(_: None = Depends(_check_auth)):
    return {
        "status": "ok",
        "tool_count": len(_TOOL_HANDLERS),
        "server": "neo4j_mcp",
        "version": "1.0.0",
        "neo4j_available": _neo4j_available,
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
        log_json("INFO", "neo4j_mcp_tool_called", details={"tool": name, "elapsed_ms": elapsed})
        return ToolResult(tool_name=name, result=result, elapsed_ms=elapsed)
    except ValueError as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("WARN", "neo4j_mcp_tool_bad_args", details={"tool": name, "error": str(exc)})
        return ToolResult(tool_name=name, error=str(exc), elapsed_ms=elapsed)
    except Exception as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        _call_errors[name] = _call_errors.get(name, 0) + 1
        log_json("ERROR", "neo4j_mcp_tool_error", details={"tool": name, "error": str(exc)})
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
    port = int(os.getenv("NEO4J_MCP_PORT", _cfg.get_mcp_server_port("neo4j", default=8013)))
    uvicorn.run("tools.neo4j_mcp:app", host="0.0.0.0", port=port, reload=False)
