"""
AURA MCP Skills Server — exposes all 20 AURA skill modules as MCP-compatible HTTP tools.

Endpoints:
  GET  /tools          → list all skills as MCP tool descriptors
  POST /call           → invoke a skill by name with args dict
  GET  /health         → health + loaded skill count
  GET  /skill/{name}   → descriptor for a single skill

Start:
  uvicorn tools.aura_mcp_skills_server:app --port 8002
  # or from project root:
  python -m uvicorn tools.aura_mcp_skills_server:app --host 0.0.0.0 --port 8002

Auth (optional):
  Set MCP_API_TOKEN env var — all requests must include Authorization: Bearer <token>
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Input/output schemas for each skill (used to build MCP tool descriptors)
# ---------------------------------------------------------------------------
_SKILL_SCHEMAS: Dict[str, Dict] = {
    "dependency_analyzer": {
        "description": "Parse requirements files in a project; detect version conflicts and known CVEs.",
        "input": {"project_root": {"type": "string", "description": "Path to project root", "default": "."}},
    },
    "architecture_validator": {
        "description": "Build import graph from .py files; detect circular imports and measure coupling score.",
        "input": {"project_root": {"type": "string", "description": "Path to project root", "default": "."}},
    },
    "complexity_scorer": {
        "description": "Compute cyclomatic complexity and nesting depth per function. Flags high-risk code.",
        "input": {
            "code": {"type": "string", "description": "Python source code to analyse (alternative to project_root)"},
            "file_path": {"type": "string", "description": "Filename label when code is provided"},
            "project_root": {"type": "string", "description": "Scan all .py files under this path"},
        },
    },
    "test_coverage_analyzer": {
        "description": "Run coverage.py (or heuristic fallback) to measure test coverage percentage.",
        "input": {
            "project_root": {"type": "string", "description": "Path to project root"},
            "min_target": {"type": "integer", "description": "Minimum acceptable coverage %", "default": 80},
        },
    },
    "doc_generator": {
        "description": "Generate docstring templates and README sections from Python source via AST.",
        "input": {
            "code": {"type": "string", "description": "Python source code"},
            "file_path": {"type": "string", "description": "Filename label"},
            "project_root": {"type": "string", "description": "Scan all .py files"},
        },
    },
    "performance_profiler": {
        "description": "Profile a code snippet with cProfile and detect AST-level anti-patterns (nested loops, re.compile in loop).",
        "input": {"code": {"type": "string", "description": "Python source code to profile"}},
    },
    "refactoring_advisor": {
        "description": "Detect code smells: god functions, deep nesting, magic numbers, long parameter lists.",
        "input": {
            "code": {"type": "string", "description": "Python source code"},
            "file_path": {"type": "string", "description": "Filename label"},
            "project_root": {"type": "string", "description": "Scan all .py files"},
        },
    },
    "schema_validator": {
        "description": "Validate a JSON schema against an instance; discover Pydantic model definitions.",
        "input": {
            "schema": {"type": "object", "description": "JSON Schema dict"},
            "instance": {"description": "Data to validate against schema"},
            "code": {"type": "string", "description": "Python source to scan for Pydantic models"},
        },
    },
    "security_scanner": {
        "description": "Static security scan: hardcoded secrets, SQL injection, unsafe eval/exec/pickle/shell=True.",
        "input": {
            "code": {"type": "string", "description": "Python source code"},
            "file_path": {"type": "string", "description": "Filename label"},
            "project_root": {"type": "string", "description": "Scan all .py files"},
        },
    },
    "type_checker": {
        "description": "Run mypy (or annotation coverage heuristic) and report type errors.",
        "input": {
            "project_root": {"type": "string", "description": "Path to project root"},
            "file_path": {"type": "string", "description": "Single file to check"},
        },
    },
    "linter_enforcer": {
        "description": "Run flake8 for style violations and AST-based naming convention checks.",
        "input": {
            "project_root": {"type": "string", "description": "Path to project root"},
            "file_path": {"type": "string", "description": "Single file to lint"},
            "code": {"type": "string", "description": "Inline code to lint"},
        },
    },
    "incremental_differ": {
        "description": "Compute unified diff between old/new code; detect added/removed symbols and impact files.",
        "input": {
            "old_code": {"type": "string", "description": "Previous version of source code"},
            "new_code": {"type": "string", "description": "New version of source code"},
            "file_path": {"type": "string", "description": "Filename label"},
            "project_root": {"type": "string", "description": "Used for impact analysis"},
        },
    },
    "tech_debt_quantifier": {
        "description": "Scan for TODO/FIXME/HACK comments, long files, long functions; compute debt score 0-100.",
        "input": {"project_root": {"type": "string", "description": "Path to project root"}},
    },
    "api_contract_validator": {
        "description": "Extract FastAPI/Flask endpoint definitions; detect breaking changes vs old spec.",
        "input": {
            "code": {"type": "string", "description": "Python source with route definitions"},
            "project_root": {"type": "string", "description": "Scan all .py files"},
            "old_spec": {"type": "object", "description": "Previous endpoint spec for diff"},
        },
    },
    "generation_quality_checker": {
        "description": "Score AI-generated code quality: syntax, structure, docstrings, intent match (no LLM needed).",
        "input": {
            "task": {"type": "string", "description": "The task the code was generated for"},
            "generated_code": {"type": "string", "description": "The generated Python source code"},
        },
    },
    "git_history_analyzer": {
        "description": "Analyse git log to identify hotspot files, change frequency patterns, and risky areas.",
        "input": {
            "project_root": {"type": "string", "description": "Path to git repository root"},
            "lookback_days": {"type": "integer", "description": "How many days of history to analyse", "default": 30},
        },
    },
    "skill_composer": {
        "description": "Map a natural-language goal to an ordered workflow of skills to execute.",
        "input": {
            "goal": {"type": "string", "description": "Natural language goal or task description"},
            "available_skills": {"type": "array", "description": "Optional list of skill names to restrict to"},
        },
    },
    "error_pattern_matcher": {
        "description": "Match a runtime error message against 12 known patterns and return fix steps.",
        "input": {
            "current_error": {"type": "string", "description": "Error message or traceback text"},
            "error_history": {"type": "array", "description": "Past {error, fix, success} records for similarity matching"},
        },
    },
    "code_clone_detector": {
        "description": "Find exact (AST-hash) and near-duplicate (Jaccard similarity) code clones across project.",
        "input": {
            "project_root": {"type": "string", "description": "Path to project root"},
            "min_lines": {"type": "integer", "description": "Minimum function size to consider", "default": 6},
            "similarity_threshold": {"type": "number", "description": "Jaccard similarity threshold (0-1)", "default": 0.8},
        },
    },
    "adaptive_strategy_selector": {
        "description": "Recommend best execution strategy based on goal type and historical success rates.",
        "input": {
            "goal": {"type": "string", "description": "Goal or task description"},
            "available_strategies": {"type": "array", "description": "List of strategy names to choose from"},
            "record_result": {"type": "object", "description": "Optional: record {strategy, success, cycles, stop_reason} to update stats"},
        },
    },
    "web_fetcher": {
        "description": "Fetch a URL and return plain text, or search DuckDuckGo and return results.",
        "input": {
            "url": {"type": "string", "description": "URL to fetch"},
            "query": {"type": "string", "description": "Search query (uses DuckDuckGo Lite)"},
            "max_chars": {"type": "integer", "description": "Max characters to return", "default": 4000},
        },
    },
    "symbol_indexer": {
        "description": "Build a project-wide symbol map: all classes, functions, methods with file/line/docstring.",
        "input": {
            "project_root": {"type": "string", "description": "Path to project root"},
            "include_private": {"type": "boolean", "description": "Include _private symbols", "default": False},
        },
    },
    "multi_file_editor": {
        "description": "Plan ordered multi-file changes for a goal, using symbol map for dependency awareness.",
        "input": {
            "goal": {"type": "string", "description": "Natural language description of the change"},
            "files": {"type": "array", "description": "Optional explicit list of files to include"},
            "symbol_map": {"type": "object", "description": "Optional output from symbol_indexer"},
            "project_root": {"type": "string", "description": "Path to project root"},
        },
    },
    "dockerfile_analyzer": {
        "description": "Analyse a Dockerfile for security issues, best-practice violations, and image hygiene (secrets in ENV, root USER, missing HEALTHCHECK, :latest tags, curl-pipe-to-shell, etc.).",
        "input": {
            "content": {"type": "string", "description": "Raw Dockerfile text to analyse (alternative to project_root)"},
            "file_path": {"type": "string", "description": "Label for the file when content is provided"},
            "project_root": {"type": "string", "description": "Scan all Dockerfiles found under this directory"},
        },
    },
    "observability_checker": {
        "description": "Check Python code for logging coverage: silent except blocks, bare print() calls, and functions with no log statements.",
        "input": {
            "code": {"type": "string", "description": "Python source code to analyse (alternative to project_root)"},
            "file_path": {"type": "string", "description": "Label for the file when code is provided"},
            "project_root": {"type": "string", "description": "Scan all .py files under this directory"},
        },
    },
    "changelog_generator": {
        "description": "Generate a structured changelog from git commit history using Conventional Commits. Returns grouped entries, version bump recommendation, and optional Markdown.",
        "input": {
            "project_root": {"type": "string", "description": "Path to the git repository root", "default": "."},
            "from_ref": {"type": "string", "description": "Start ref (tag, SHA, or branch). Defaults to all commits."},
            "to_ref": {"type": "string", "description": "End ref (default: HEAD)"},
            "limit": {"type": "integer", "description": "Max commits to include (default 100)"},
            "include_markdown": {"type": "boolean", "description": "Include formatted Markdown in output (default true)"},
        },
    },
    "database_query_analyzer": {
        "description": "Detect database query anti-patterns in Python code: N+1 queries, SELECT *, leading-wildcard LIKE, deep OFFSET pagination, SQL injection risks, destructive DDL, and more.",
        "input": {
            "code": {"type": "string", "description": "Python source code to analyse (alternative to project_root)"},
            "file_path": {"type": "string", "description": "Label for the file when code is provided"},
            "project_root": {"type": "string", "description": "Scan all .py files under this directory"},
        },
    },
}


# ---------------------------------------------------------------------------
# App + auth
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AURA MCP Skills Server",
    description="Exposes all AURA software-engineering skills as MCP-compatible HTTP tools.",
    version="1.0.0",
)

_MCP_TOKEN = os.getenv("MCP_API_TOKEN")
_skills: Dict[str, Any] = {}
_load_error: Optional[str] = None


def _load_skills() -> None:
    global _skills, _load_error
    try:
        from agents.skills.registry import all_skills
        _skills = all_skills()
        log_json("INFO", "mcp_skills_server_loaded", details={"count": len(_skills)})
    except Exception as exc:
        _load_error = str(exc)
        log_json("ERROR", "mcp_skills_server_load_failed", details={"error": str(exc)})


_load_skills()


def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not _MCP_TOKEN:
        return
    if not authorization or authorization != f"Bearer {_MCP_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing Authorization header")


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class CallRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health(_: None = Depends(require_auth)) -> Dict:
    return {
        "status": "ok" if not _load_error else "degraded",
        "skills_loaded": len(_skills),
        "load_error": _load_error,
        "server": "aura_mcp_skills_server",
        "version": "1.0.0",
    }


@app.get("/tools")
async def list_tools(_: None = Depends(require_auth)) -> Dict:
    """Return MCP-compatible tool list for all loaded skills."""
    tools: List[Dict] = []
    for name, skill in _skills.items():
        schema = _SKILL_SCHEMAS.get(name, {})
        tools.append({
            "name": name,
            "description": schema.get("description", f"AURA skill: {name}"),
            "inputSchema": {
                "type": "object",
                "properties": schema.get("input", {}),
            },
        })
    # Add any skills loaded but not in schema dict
    for name in _skills:
        if not any(t["name"] == name for t in tools):
            tools.append({"name": name, "description": f"AURA skill: {name}", "inputSchema": {"type": "object", "properties": {}}})
    return {"tools": tools, "count": len(tools)}


@app.get("/skill/{name}")
async def get_skill(name: str, _: None = Depends(require_auth)) -> Dict:
    """Return descriptor for a single skill."""
    if name not in _skills:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found. Available: {sorted(_skills.keys())}")
    schema = _SKILL_SCHEMAS.get(name, {})
    return {
        "name": name,
        "description": schema.get("description", f"AURA skill: {name}"),
        "inputSchema": {"type": "object", "properties": schema.get("input", {})},
        "skill_class": type(_skills[name]).__name__,
    }


@app.post("/call")
async def call_skill(req: CallRequest, _: None = Depends(require_auth)) -> Dict:
    """Invoke a skill by name with the given args dict."""
    if _load_error and not _skills:
        raise HTTPException(status_code=503, detail=f"Skills failed to load: {_load_error}")

    skill = _skills.get(req.tool_name)
    if skill is None:
        raise HTTPException(
            status_code=404,
            detail=f"Skill '{req.tool_name}' not found. Available: {sorted(_skills.keys())}",
        )

    t0 = time.perf_counter()
    try:
        result = skill.run(req.args)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        log_json("INFO", "mcp_skill_called", details={"skill": req.tool_name, "elapsed_ms": elapsed_ms})
        return {"status": "success", "skill": req.tool_name, "result": result, "elapsed_ms": elapsed_ms}
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        log_json("ERROR", "mcp_skill_error", details={"skill": req.tool_name, "error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
async def root() -> Dict:
    return {
        "server": "AURA MCP Skills Server",
        "version": "1.0.0",
        "skills": sorted(_skills.keys()),
        "endpoints": ["/tools", "/call", "/skill/{name}", "/health"],
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MCP_SKILLS_PORT", "8002"))
    log_json("INFO", "mcp_skills_server_starting", details={"port": port, "skills": len(_skills)})
    uvicorn.run("tools.aura_mcp_skills_server:app", host="0.0.0.0", port=port, reload=False)
