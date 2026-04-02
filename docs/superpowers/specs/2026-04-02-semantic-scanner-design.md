# Semantic Codebase Scanner — Design Spec

**Status:** Approved
**Date:** 2026-04-02
**Builds on:** `core/agent_sdk/` production loop (12 modules, 106 tests)

## Overview

A three-layer semantic scanner that builds a SQLite index of the codebase with LLM-augmented intent summaries. The index feeds into the agent-sdk workflow at two points: a compact architectural overview injected into the system prompt at session start, and a `query_codebase` tool for targeted deep dives during execution. Incremental refresh via git diff ensures the index stays current without re-scanning unchanged files.

## 1. Scanner Architecture

Three-layer pipeline producing a SQLite index:

### Layer 1: AST Extraction (no LLM)

Uses Python's `ast` module to extract:
- **Symbols:** Functions, classes, methods — name, kind, line number, signature, docstring, decorators
- **Class inheritance:** For each class, extract base classes from `ast.ClassDef.bases` — stores as `(class_symbol_id, base_class_name)` pairs
- **Imports:** Module-level imports — what each file imports and from where
- **Call sites:** Function/method calls within each function body — caller, callee name, line

Scans all `.py` files in project directories, excluding: `.git/`, `__pycache__/`, `node_modules/`, `.venv/`, `venv/`, `*.egg-info/`.

### Layer 2: Relationship Analysis (no LLM)

Builds from Layer 1 data:
- **Call graph:** Directed edges from caller symbols to callee symbols (resolved via imports where possible)
- **Dependency chains:** File-level import graph — which files depend on which
- **Coupling scores:** Per-file metric: `(inbound_edges + outbound_edges) / total_files` normalized to 0-1
- **Module clusters:** Group files by directory prefix (e.g., `aura_cli/`, `core/`, `agents/`, `tools/`)

### Layer 3: Intent Augmentation (LLM via ModelAdapter)

Generates natural-language summaries:
- **Module summary:** 2-3 sentence description of what each file does and its role in the system
- **Function intent:** 1 sentence per function (>10 lines only) describing what it does and why

**Cost control:**
- Skip files < 5 lines
- Skip functions < 10 lines
- Use cheapest model (`claude-haiku-4-5` or local model)
- Batch up to 10 functions per LLM call
- Budget cap: `config.scan_llm_budget_usd` (default $0.50)
- Estimated cost for AURA (~200 files): ~$0.15 with Haiku

**LLM prompts:**

Module-level:
```
Summarize this Python module in 2-3 sentences. Focus on: what it does,
what role it plays in the system, and what other modules depend on it.
Module: {file_path}
Symbols: {symbol_list_with_signatures}
Imports: {import_list}
Called by: {callers_from_layer2}
```

Function-level (batched):
```
For each function below, write ONE sentence describing what it does and why.
Format: function_name: description

{function_1_name}({signature})
{body_truncated_500_chars}

{function_2_name}({signature})
{body_truncated_500_chars}
...
```

**Graceful degradation when no LLM available:**
- `intent_summary` stays NULL in DB
- `summarize()` returns docstring instead
- `find_similar()` falls back to keyword matching on names + docstrings
- Layers 1+2 work perfectly without LLM

## 2. Storage Schema

File: `memory/semantic_index.db` (SQLite, WAL mode, CREATE TABLE IF NOT EXISTS on init)

### files table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| path | TEXT UNIQUE | Relative path from project root |
| module_name | TEXT | Dotted module name (e.g., `core.agent_sdk.config`) |
| cluster | TEXT | Directory prefix (e.g., `core/agent_sdk`) |
| line_count | INTEGER | Total lines |
| last_modified | TEXT | File mtime ISO timestamp |
| last_scan_sha | TEXT | Git SHA when file was last scanned |
| module_summary | TEXT | LLM-generated module description (nullable) |
| coupling_score | REAL | 0-1 coupling metric |
| scanned_at | TEXT | ISO timestamp of last scan |

### symbols table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| file_id | INTEGER FK | References files.id |
| name | TEXT | Symbol name |
| kind | TEXT | function, class, method, classmethod, staticmethod, property |
| line_start | INTEGER | Start line |
| line_end | INTEGER | End line |
| signature | TEXT | Parameter signature string |
| docstring | TEXT | First docstring (nullable) |
| decorators | TEXT | Comma-separated decorator names |
| intent_summary | TEXT | LLM-generated intent (nullable) |

### imports table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| file_id | INTEGER FK | References files.id |
| imported_module | TEXT | e.g., `core.skill_dispatcher` |
| imported_name | TEXT | e.g., `SKILL_MAP` (nullable for full-module imports) |
| is_from_import | BOOLEAN | True for `from X import Y` |

### call_sites table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| caller_symbol_id | INTEGER FK | References symbols.id |
| callee_name | TEXT | Name of called function/method |
| line | INTEGER | Line number of call |

### relationships table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| from_file_id | INTEGER FK | References files.id |
| to_file_id | INTEGER FK | References files.id |
| rel_type | TEXT | imports, calls, inherits |
| strength | REAL | 0-1 normalized weight |

### symbols_fts virtual table (FTS5)

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name, docstring, intent_summary, content=symbols, content_rowid=id
);
```

Populated/updated whenever symbols are inserted or updated. Used by `find_similar()` for full-text search.

### scan_meta table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| scan_sha | TEXT | Git HEAD SHA at scan time |
| scan_time | TEXT | ISO timestamp |
| files_scanned | INTEGER | Count |
| symbols_found | INTEGER | Count |
| llm_calls_made | INTEGER | Count |
| llm_cost_usd | REAL | Total LLM cost |
| scan_type | TEXT | full, incremental |

## 3. Incremental Refresh

### Trigger

`scanner.refresh_if_needed()` called by `ContextBuilder.build()` before assembling context.

### Algorithm

```
1. Get current HEAD SHA: git rev-parse HEAD
2. Load last_scan_sha from scan_meta
3. If same → no-op (return immediately, ~1ms)
4. If different:
   a. Try: git diff --name-only {last_scan_sha}..HEAD -- '*.py'
      If last_scan_sha is not a valid ref (orphaned after rebase/GC):
        → fall back to full scan (log warning)
   b. For each changed file:
      - Re-run Layer 1 (AST extraction) → update symbols, imports, call_sites
      - Re-run Layer 2 (relationship analysis) → update relationships, coupling_scores
      - If LLM available and file changed significantly (>20% line diff):
        Re-run Layer 3 (intent summaries) → update module_summary, intent_summary
   c. Delete entries for deleted files
   d. Insert entries for new files (full Layer 1+2+3)
   e. Record new scan_meta entry
   f. Run PRAGMA wal_checkpoint(TRUNCATE) after full scans to prevent WAL growth
```

### Performance

- No-op check: ~1ms (single SQL query)
- Incremental (5 changed files): ~2-5 seconds (AST + relationships) + ~$0.02 LLM
- Full scan (~200 files): ~10 seconds + ~$0.15 LLM

## 4. Query Interface

### SemanticQuerier class

File: `core/agent_sdk/semantic_querier.py`

```python
class SemanticQuerier:
    def __init__(self, db_path: Path)
    
    def what_calls(self, symbol_name: str) -> List[Dict]
        """All callers of a symbol. Returns [{caller, file, line}]"""
    
    def what_depends_on(self, file_path: str) -> List[Dict]
        """Files that import from this file. Returns [{path, imported_names}]"""
    
    def what_changes_break(self, file_path: str, depth: int = 2) -> List[Dict]
        """Transitive dependents (ripple analysis). Returns [{path, distance, relationship}]"""
    
    def summarize(self, target: str) -> str
        """Return LLM summary for a file or symbol. Falls back to docstring."""
    
    def find_similar(self, description: str, limit: int = 5) -> List[Dict]
        """Find symbols matching a natural-language description.
        Uses SQLite FTS5 full-text search on intent_summary + docstring + name.
        When no LLM summaries exist, falls back to FTS on names + docstrings only.
        Note: This is keyword/token matching, not semantic embedding search.
        Good for explicit terms ('auth', 'retry', 'error handler') but weak for
        conceptual queries ('something that prevents infinite loops'). For deeper
        semantic search, use AURA's existing vector_store if available."""
    
    def architecture_overview(self) -> Dict
        """Module clusters, top coupled files, key boundaries. ~500 token summary."""
    
    def recent_changes(self, n_commits: int = 5) -> List[Dict]
        """Changed files + their summaries since N commits ago."""
```

### MCP Tool Registration

Registered in `tool_registry.py` as `query_codebase`:

```python
{
    "name": "query_codebase",
    "description": "Query the semantic codebase index for deep code understanding. "
                   "Use this to understand call chains, dependencies, impact of changes, "
                   "and to find relevant code by description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": ["what_calls", "what_depends_on", "what_changes_break",
                         "summarize", "find_similar", "architecture_overview", 
                         "recent_changes"],
            },
            "target": {"type": "string", "description": "File path, symbol name, or description"},
            "depth": {"type": "integer", "default": 2, "description": "Recursion depth for transitive queries"},
        },
        "required": ["query_type"],
    },
}
```

## 5. Context Injection into Agent-SDK

### ContextBuilder Enhancement

`ContextBuilder.build()` gains a new method `_get_codebase_context(goal)`:

```python
def _get_codebase_context(self, goal: str) -> Dict[str, Any]:
    """Query semantic index for goal-relevant codebase context."""
    if self._semantic_querier is None:
        return {}
    
    overview = self._semantic_querier.architecture_overview()
    relevant = self._semantic_querier.find_similar(goal, limit=5)
    
    # Extract file paths mentioned in goal for impact analysis
    goal_files = [s["file"] for s in relevant if "file" in s]
    impact = []
    for f in goal_files[:3]:  # limit to avoid prompt bloat
        deps = self._semantic_querier.what_depends_on(f)
        impact.extend(deps)
    
    return {
        "codebase_overview": overview,
        "relevant_symbols": relevant,
        "impact_radius": impact,
    }
```

Returns are added to the context dict and rendered in `build_system_prompt()`:

```
### Codebase Understanding
**Architecture:** {overview summary}

**Relevant Code:**
{for each relevant symbol: - file:line: name — intent_summary}

**Impact Radius:** Changes to {files} affect: {dependents} ({count} direct, {count} transitive)
```

### ContextBuilder Constructor Change

`ContextBuilder.__init__` gains `semantic_querier` as a **fourth** optional keyword argument, after the existing `vector_store`:

```python
def __init__(
    self,
    project_root: Path,
    brain: Any = None,
    vector_store: Any = None,       # existing — preserved
    semantic_querier: Any = None,   # NEW — optional
) -> None:
```

When None, `_get_codebase_context()` returns empty dict — fully backward compatible. Existing callers that pass `(project_root, brain)` or `(project_root, brain, vector_store)` are unaffected.

The controller creates the querier from config. **Important:** The existing `_build_prompt()` method creates a fresh `ContextBuilder` internally — it must be changed to use `self.context_builder` instead, so the querier is available for prompt generation.

```python
# In AuraController.__init__ (after existing context_builder creation):
querier = None
if config.semantic_index_path.exists():
    from core.agent_sdk.semantic_querier import SemanticQuerier
    querier = SemanticQuerier(db_path=config.semantic_index_path)

self.context_builder = ContextBuilder(
    project_root=project_root, brain=brain,
    semantic_querier=querier,
)

# In AuraController._build_prompt() — change from creating fresh builder to:
def _build_prompt(self, goal: str) -> str:
    context = self.context_builder.build(goal=goal)
    return self.context_builder.build_system_prompt(
        goal=goal, goal_type=context["goal_type"], context=context,
    )
```

### Tool Handler + Dependency Threading

The `query_codebase` tool handler dispatches `query_type` to `SemanticQuerier` methods:

```python
def _handle_query_codebase(
    args: Dict[str, Any],
    *,
    semantic_querier: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Dispatch query_type to SemanticQuerier methods."""
    if semantic_querier is None:
        return {"error": "Semantic index not available. Run 'agent scan' first."}
    
    query_type = args["query_type"]
    target = args.get("target", "")
    depth = args.get("depth", 2)
    
    dispatch = {
        "what_calls": lambda: semantic_querier.what_calls(target),
        "what_depends_on": lambda: semantic_querier.what_depends_on(target),
        "what_changes_break": lambda: semantic_querier.what_changes_break(target, depth=depth),
        "summarize": lambda: {"summary": semantic_querier.summarize(target)},
        "find_similar": lambda: {"results": semantic_querier.find_similar(target)},
        "architecture_overview": lambda: semantic_querier.architecture_overview(),
        "recent_changes": lambda: {"changes": semantic_querier.recent_changes(n_commits=depth)},
    }
    handler = dispatch.get(query_type)
    if not handler:
        return {"error": f"Unknown query_type: {query_type}"}
    try:
        return handler()
    except Exception as exc:
        return {"error": str(exc)}
```

`create_aura_tools()` gains an optional `semantic_querier` parameter. Full updated signature:

```python
def create_aura_tools(
    project_root: Path,
    brain: Any = None,
    model_adapter: Any = None,
    goal_queue: Any = None,
    goal_archive: Any = None,
    config: Any = None,
    semantic_querier: Any = None,   # NEW
) -> List[AuraTool]:
```

The `semantic_querier` is added to the `deps` dict and bound to `_handle_query_codebase` via the existing closure pattern. The `query_codebase` tool is added to the `_TOOL_DEFS` list with `"handler": _handle_query_codebase`.

The controller passes its querier instance:

```python
# In AuraController._build_mcp_server():
tools = create_aura_tools(
    project_root=self.project_root,
    brain=self._brain,
    model_adapter=self._model_adapter,
    goal_queue=self._goal_queue,
    goal_archive=self._goal_archive,
    config=self.config,
    semantic_querier=querier,
)
```

## 6. CLI Commands

| Command | Action |
|---------|--------|
| `agent scan` | Run full semantic scan (all 3 layers) |
| `agent scan --incremental` | Incremental refresh only (default on `agent run`) |
| `agent scan --no-llm` | Layers 1+2 only, skip intent augmentation |
| `agent scan --stats` | Show scan metadata (files, symbols, cost, last scan time) |

## 7. Config Additions

New fields on `AgentSDKConfig`:

```python
semantic_index_path: Path = Path("memory/semantic_index.db")
scan_llm_budget_usd: float = 0.50
scan_llm_model: str = "claude-haiku-4-5"  # cheapest for summaries
scan_exclude_patterns: List[str] = field(default_factory=lambda: [
    ".git", "__pycache__", "node_modules", ".venv", "venv", "*.egg-info",
])
scan_min_function_lines: int = 10  # skip small functions for LLM summaries
scan_min_file_lines: int = 5       # skip trivial files for LLM summaries
scan_batch_size: int = 10           # functions per LLM call
```

`from_aura_config()` must be updated to add these lines in the `cls(...)` call:

```python
            semantic_index_path=Path(sdk_section.get("semantic_index_path", "memory/semantic_index.db")),
            scan_llm_budget_usd=sdk_section.get("scan_llm_budget_usd", 0.50),
            scan_llm_model=sdk_section.get("scan_llm_model", "claude-haiku-4-5"),
            scan_exclude_patterns=sdk_section.get("scan_exclude_patterns", [
                ".git", "__pycache__", "node_modules", ".venv", "venv", "*.egg-info",
            ]),
            scan_min_function_lines=sdk_section.get("scan_min_function_lines", 10),
            scan_min_file_lines=sdk_section.get("scan_min_file_lines", 5),
            scan_batch_size=sdk_section.get("scan_batch_size", 10),
```

**LLM routing:** The scanner uses `ModelAdapter.respond_for_role(route_key, prompt)` — not `respond()` which takes only a prompt string. The route_key is set to `"code_generation"` (which maps to the cheapest configured model in `settings.json` model_routing). This uses the existing model routing infrastructure without requiring a new parameter.

```python
# In semantic_scanner.py Layer 3:
summary = model_adapter.respond_for_role("code_generation", prompt)
```

If `ModelAdapter` is not available, Layer 3 degrades gracefully (no summaries generated). The `scan_llm_model` config field is reserved for future use when `ModelAdapter` supports explicit model override — for now it serves as documentation of intent.

## 8. File Structure

```
core/agent_sdk/
├── semantic_scanner.py       # NEW: Three-layer scanner pipeline
├── semantic_querier.py       # NEW: Query interface for the index
├── semantic_schema.py        # NEW: SQLite schema + data access layer
├── context_builder.py        # MODIFY: Add _get_codebase_context()
├── controller.py             # MODIFY: Create querier, pass to context_builder
├── tool_registry.py          # MODIFY: Add query_codebase tool
├── config.py                 # MODIFY: Add scan config fields
├── cli_integration.py        # MODIFY: Add agent scan handler
└── (existing unchanged)

tests/
├── test_semantic_scanner.py      # AST extraction, relationship analysis
├── test_semantic_querier.py      # All 7 query types
├── test_semantic_schema.py       # SQLite CRUD
├── test_semantic_integration.py  # Full scan + query pipeline
```

## 9. Dependencies

- `ast` (stdlib) — Python AST parsing
- `sqlite3` (stdlib) — Index storage
- `gitpython` (already in requirements.txt) — Git operations for incremental refresh
- `core.model_adapter.ModelAdapter` (existing) — LLM calls for Layer 3
- No new external dependencies required

## 10. Testing Strategy

- **Layer 1 tests:** Parse known Python files, verify symbols/imports/call_sites extracted correctly
- **Layer 2 tests:** Build relationships from Layer 1 fixtures, verify call graph and coupling scores
- **Layer 3 tests:** Mock ModelAdapter, verify LLM prompts are correctly constructed and summaries stored
- **Querier tests:** Pre-populate SQLite with fixtures, test all 7 query types
- **Incremental tests:** Modify a file, run refresh, verify only changed file re-scanned
- **Integration test:** Full scan of a small test fixture directory, then query and verify results
- **Graceful degradation:** Test Layer 3 with no LLM available — verify Layers 1+2 still work
