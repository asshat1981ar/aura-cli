"""Semantic MCP Discovery Server for AURA.

Aggregates tool manifests from all running MCP servers in the cluster
and provides heuristic semantic search using token overlap.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_discovery")

app = FastAPI(title="AURA Discovery MCP", version="0.1.0")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "aura.config.json")
API_TOKEN = os.getenv("MCP_API_TOKEN", "default_token")
DISCOVERY_PORT = int(os.getenv("AURA_DISCOVERY_PORT", "8025"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class CallRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------


def get_mcp_config() -> Dict[str, int]:
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
            return config.get("mcp_servers", {})
    except Exception as e:
        logger.error(f"Failed to read aura.config.json: {e}")
        return {}


def _extract_tools_payload(payload: Any) -> List[Dict[str, Any]]:
    """Accept both list and wrapped MCP payload formats."""
    if isinstance(payload, list):
        return [tool for tool in payload if isinstance(tool, dict)]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            tools = data.get("tools", [])
            if isinstance(tools, list):
                return [tool for tool in tools if isinstance(tool, dict)]
    return []


def _normalize_tool(tool: Dict[str, Any], server_name: str, port: int) -> Dict[str, Any]:
    """Return a stable discovery record for a tool."""
    return {
        "name": tool.get("name", "unknown"),
        "description": tool.get("description", ""),
        "server": tool.get("server", server_name),
        "port": tool.get("port", port),
    }


def list_all_tools() -> List[Dict[str, Any]]:
    servers = get_mcp_config()
    all_tools: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()

    for name, port in servers.items():
        # Skip ourselves to avoid infinite loops if we were registered
        if name == "discovery" or port == DISCOVERY_PORT:
            continue

        url = f"http://localhost:{port}/tools"
        try:
            # Note: We use a short timeout for discovery
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                timeout=0.5,
            )
            if response.status_code == 200:
                tools = _extract_tools_payload(response.json())
                for tool in tools:
                    normalized = _normalize_tool(tool, name, port)
                    key = (
                        normalized["server"],
                        normalized["name"],
                        normalized["port"],
                    )
                    if key not in seen:
                        seen.add(key)
                        all_tools.append(normalized)
        except Exception as e:
            logger.debug(f"Server {name} at port {port} is offline or unreachable: {e}")

    return all_tools


def score_match(query: str, tool: Dict[str, Any]) -> float:
    """Simple similarity score based on token overlap."""
    query_tokens = set(query.lower().split())
    if not query_tokens:
        return 0.0
    name_tokens = set(tool.get("name", "").lower().replace("_", " ").split())
    desc_tokens = set(tool.get("description", "").lower().split())

    # Combined target tokens
    target_tokens = name_tokens.union(desc_tokens)

    if not target_tokens:
        return 0.0

    intersection = query_tokens.intersection(target_tokens)
    # Jaccard-ish similarity with name weighting
    name_intersection = query_tokens.intersection(name_tokens)

    score = (len(intersection) / len(query_tokens.union(target_tokens))) + (0.5 if name_intersection else 0.0)
    return score


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/tools")
async def tools_list():
    return {"data": {"tools": [{"name": "list_all_mcp_tools", "description": "Aggregate tool lists from all running MCP servers."}, {"name": "search_tools_semantically", "description": "Search for tools matching a purpose or goal description."}]}}


@app.post("/call")
async def call_tool(req: CallRequest):
    name = req.tool_name
    args = req.args

    if name == "list_all_mcp_tools":
        tools = list_all_tools()
        return {"data": {"tools": tools, "count": len(tools)}}

    if name == "search_tools_semantically":
        query = args.get("query", "")
        top_k = int(args.get("top_k", 5))

        all_tools = list_all_tools()
        scored = []
        for t in all_tools:
            score = score_match(query, t)
            if score > 0:
                scored.append((score, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, tool in scored[:top_k]:
            results.append({**tool, "score": round(score, 4)})
        return {"data": {"results": results, "query": query, "count": len(results)}}

    raise HTTPException(status_code=404, detail=f"Unknown tool: {name}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=DISCOVERY_PORT)
