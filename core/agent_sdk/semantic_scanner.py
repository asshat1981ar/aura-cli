"""Three-layer semantic codebase scanner.

Layer 1 (AST):   Extract symbols, imports, call sites from Python files.
Layer 2 (Graph): Build file-level import relationships, compute coupling scores.
Layer 3 (LLM):   Generate module/function summaries via ModelAdapter (optional).
"""
from __future__ import annotations

import ast
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.agent_sdk.semantic_schema import SemanticDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal AST helpers
# ---------------------------------------------------------------------------

def _decorator_name(node: ast.expr) -> str:
    """Return a string name for a decorator node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_decorator_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return "<decorator>"


def _base_name(node: ast.expr) -> str:
    """Return a string name for a base-class node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_base_name(node.value)}.{node.attr}"
    return "<base>"


def _call_name(node: ast.expr) -> Optional[str]:
    """Return a string name for the callable in a Call node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _arg_annotation(arg: ast.arg) -> str:
    if arg.annotation is None:
        return ""
    try:
        return ast.unparse(arg.annotation)
    except (ValueError, TypeError):
        return ""


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a compact signature string from a function node."""
    parts: List[str] = []
    args = node.args

    # positional-only (Python 3.8+)
    for a in args.posonlyargs:
        ann = _arg_annotation(a)
        parts.append(f"{a.arg}: {ann}" if ann else a.arg)
    if args.posonlyargs:
        parts.append("/")

    defaults_offset = len(args.args) - len(args.defaults)
    for i, a in enumerate(args.args):
        ann = _arg_annotation(a)
        param = f"{a.arg}: {ann}" if ann else a.arg
        di = i - defaults_offset
        if di >= 0:
            try:
                param += f" = {ast.unparse(args.defaults[di])}"
            except (ValueError, TypeError):
                param += " = ..."
        parts.append(param)

    if args.vararg:
        a = args.vararg
        ann = _arg_annotation(a)
        parts.append(f"*{a.arg}: {ann}" if ann else f"*{a.arg}")
    elif args.kwonlyargs:
        parts.append("*")

    for a in args.kwonlyargs:
        ann = _arg_annotation(a)
        parts.append(f"{a.arg}: {ann}" if ann else a.arg)

    if args.kwarg:
        a = args.kwarg
        ann = _arg_annotation(a)
        parts.append(f"**{a.arg}: {ann}" if ann else f"**{a.arg}")

    sig = f"{node.name}({', '.join(parts)})"
    if node.returns:
        try:
            sig += f" -> {ast.unparse(node.returns)}"
        except (ValueError, TypeError):
            pass
    return sig


def _parse_tree(file_path: Path) -> Optional[ast.Module]:
    """Parse a file and return the AST, or None on SyntaxError/UnicodeDecodeError."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError):
        return None
    try:
        return ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return None


# ---------------------------------------------------------------------------
# Layer 1 — AST extraction (module-level functions)
# ---------------------------------------------------------------------------

def extract_symbols(file_path: Path) -> List[Dict[str, Any]]:
    """Extract functions, classes, and methods from a Python file.

    Returns a list of dicts with keys:
        name, kind, line_start, line_end, signature, docstring, decorators, bases
    kind values: function, class, method, staticmethod, classmethod, property
    """
    tree = _parse_tree(file_path)
    if tree is None:
        return []

    # Build a set of class-owned function nodes for method detection
    class_children: set[int] = set()
    class_decorator_map: dict[int, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_children.add(id(item))
                    class_decorator_map[id(item)] = [_decorator_name(d) for d in item.decorator_list]

    results: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node) or None
            bases = [_base_name(b) for b in node.bases]
            results.append({
                "name": node.name,
                "kind": "class",
                "line_start": node.lineno,
                "line_end": node.end_lineno or node.lineno,
                "signature": node.name,
                "docstring": docstring,
                "decorators": [_decorator_name(d) for d in node.decorator_list],
                "bases": bases,
            })

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node) or None
            decorators = [_decorator_name(d) for d in node.decorator_list]

            if id(node) in class_children:
                # Determine the method subkind from decorator names
                if "staticmethod" in decorators:
                    kind = "staticmethod"
                elif "classmethod" in decorators:
                    kind = "classmethod"
                elif "property" in decorators:
                    kind = "property"
                else:
                    kind = "method"
            else:
                kind = "function"

            results.append({
                "name": node.name,
                "kind": kind,
                "line_start": node.lineno,
                "line_end": node.end_lineno or node.lineno,
                "signature": _build_signature(node),
                "docstring": docstring,
                "decorators": decorators,
                "bases": [],
            })

    return results


def extract_imports(file_path: Path) -> List[Dict[str, Any]]:
    """Extract import statements from a Python file.

    Returns a list of dicts with keys:
        imported_module, imported_name, is_from_import
    """
    tree = _parse_tree(file_path)
    if tree is None:
        return []

    results: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append({
                    "imported_module": alias.name,
                    "imported_name": alias.asname,
                    "is_from_import": False,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                results.append({
                    "imported_module": module,
                    "imported_name": alias.name,
                    "is_from_import": True,
                })
    return results


def extract_call_sites(file_path: Path, symbol: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract function calls within a symbol's line range.

    Returns a list of dicts with keys: callee_name, line
    """
    tree = _parse_tree(file_path)
    if tree is None:
        return []

    line_start = symbol.get("line_start", 0)
    line_end = symbol.get("line_end", 0)

    results: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            line = getattr(node, "lineno", None)
            if line is None:
                continue
            if line_start <= line <= line_end:
                name = _call_name(node.func)
                if name:
                    results.append({"callee_name": name, "line": line})

    return results


# ---------------------------------------------------------------------------
# Layer 2 — Relationship analysis (module-level functions)
# ---------------------------------------------------------------------------

def build_relationships(db: SemanticDB) -> None:
    """Build file-level import relationships by resolving module names to file paths.

    For each file's imports, attempt to resolve the imported_module to a known
    file path in the DB and upsert an 'imports' relationship.
    """
    all_files = db.get_all_files()
    # Build lookup: module_name -> file_id
    module_to_id: Dict[str, int] = {}
    path_to_id: Dict[str, int] = {}
    for f in all_files:
        if f.get("module_name"):
            module_to_id[f["module_name"]] = f["id"]
        path_to_id[f["path"]] = f["id"]

    for f in all_files:
        from_id = f["id"]
        imports = db.get_imports_for_file(from_id)
        seen: set[int] = set()
        for imp in imports:
            mod = imp.get("imported_module", "")
            if not mod:
                continue
            to_id = module_to_id.get(mod)
            if to_id is None:
                # Try last component as simple filename: e.g. "pkg.foo" -> "pkg/foo.py"
                candidate_path = mod.replace(".", "/") + ".py"
                to_id = path_to_id.get(candidate_path)
            if to_id is not None and to_id != from_id and to_id not in seen:
                seen.add(to_id)
                db.upsert_relationship(from_id, to_id, "imports", 1.0)


def compute_coupling_scores(db: SemanticDB) -> None:
    """Compute coupling score for each file.

    coupling = min((inbound + outbound) / total_files, 1.0)
    """
    all_files = db.get_all_files()
    total = len(all_files)
    if total == 0:
        return

    for f in all_files:
        fid = f["id"]
        outbound = len(db.get_relationships_from(fid, rel_type="imports"))
        inbound = len(db.get_relationships_to(fid, rel_type="imports"))
        score = min((inbound + outbound) / total, 1.0)
        db.update_file_coupling(fid, score)


# ---------------------------------------------------------------------------
# Layer 3 — LLM enrichment (module-level functions)
# ---------------------------------------------------------------------------

def generate_module_summary(
    db: SemanticDB,
    file_id: int,
    file_path: Path,
    model_adapter: Any,
) -> Optional[str]:
    """Generate an LLM summary for a module. Returns None on failure."""
    try:
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        # Truncate very large files to keep prompts manageable
        if len(source) > 8000:
            source = source[:8000] + "\n... (truncated)"
        prompt = (
            f"Summarise this Python module in 1-2 sentences. "
            f"Focus on purpose and key abstractions.\n\n```python\n{source}\n```"
        )
        summary = model_adapter.respond(prompt)
        if summary:
            db.update_file_summary(file_id, summary)
        return summary
    except (OSError, RuntimeError, ValueError, ConnectionError, TimeoutError) as exc:
        logger.debug("generate_module_summary failed: %s", exc)
        return None


def generate_function_intents(
    symbols: List[Dict[str, Any]],
    file_path: Path,
    model_adapter: Any,
    batch_size: int = 10,
) -> Dict[int, str]:
    """Generate batched LLM intent summaries for symbols.

    Returns a dict mapping symbol _db_id -> summary string.
    """
    results: Dict[int, str] = {}
    if not symbols:
        return results

    # Read source once for context
    try:
        source_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        source_lines = []

    for batch_start in range(0, len(symbols), batch_size):
        batch = symbols[batch_start: batch_start + batch_size]
        # Build a prompt listing all functions in this batch
        entries: List[str] = []
        for sym in batch:
            name = sym.get("name", "?")
            sig = sym.get("signature", "")
            doc = sym.get("docstring") or ""
            start = sym.get("line_start", 1) - 1
            end = sym.get("line_end", start + 1)
            snippet = "\n".join(source_lines[start:end])[:500] if source_lines else ""
            entries.append(
                f"Function: {name}\nSignature: {sig}\nDocstring: {doc}\nBody snippet:\n{snippet}"
            )

        prompt = (
            "For each function below, write one sentence describing its intent. "
            "Reply in the format 'FunctionName: intent.' on separate lines.\n\n"
            + "\n---\n".join(entries)
        )
        try:
            response = model_adapter.respond(prompt)
            # Parse "Name: intent." lines
            intent_map: Dict[str, str] = {}
            for line in response.splitlines():
                if ":" in line:
                    left, _, right = line.partition(":")
                    intent_map[left.strip()] = right.strip()
            for sym in batch:
                db_id = sym.get("_db_id")
                if db_id is not None:
                    name = sym.get("name", "")
                    summary = intent_map.get(name, "")
                    if not summary:
                        # fallback: store the raw response for first symbol in batch
                        summary = response.strip()
                    results[db_id] = summary
        except (OSError, RuntimeError, ValueError, ConnectionError, TimeoutError) as exc:
            logger.debug("generate_function_intents batch failed: %s", exc)
            # Fill with empty strings so callers know the IDs were processed
            for sym in batch:
                db_id = sym.get("_db_id")
                if db_id is not None:
                    results[db_id] = ""

    return results


# ---------------------------------------------------------------------------
# SemanticScanner class
# ---------------------------------------------------------------------------

_DEFAULT_EXCLUDE = [
    "__pycache__",
    ".git",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "*.egg-info",
]


class SemanticScanner:
    """Orchestrates the three-layer pipeline over a project."""

    def __init__(
        self,
        project_root: Path,
        db_path: Path,
        exclude_patterns: Optional[List[str]] = None,
        model_adapter: Any = None,
        min_function_lines: int = 10,
        min_file_lines: int = 5,
        batch_size: int = 10,
        llm_budget: float = 0.50,
    ):
        self.project_root = Path(project_root)
        self.db_path = Path(db_path)
        self.exclude_patterns = list(exclude_patterns or _DEFAULT_EXCLUDE)
        self.model_adapter = model_adapter
        self.min_function_lines = min_function_lines
        self.min_file_lines = min_file_lines
        self.batch_size = batch_size
        self.llm_budget = llm_budget

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_py_files(self) -> List[Path]:
        """Find all .py files under project_root, respecting exclude_patterns."""
        found: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.project_root):
            # Prune excluded directories in-place
            dirnames[:] = [
                d for d in dirnames
                if not self._is_excluded(d)
            ]
            for fname in filenames:
                if fname.endswith(".py"):
                    found.append(Path(dirpath) / fname)
        return found

    def _is_excluded(self, name: str) -> bool:
        import fnmatch
        for pat in self.exclude_patterns:
            if fnmatch.fnmatch(name, pat):
                return True
        return False

    def _relative(self, path: Path) -> str:
        """Return path relative to project_root as a POSIX string."""
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _module_name(self, rel_path: str) -> str:
        """Convert 'core/foo.py' -> 'core.foo'."""
        return rel_path.removesuffix(".py").replace("/", ".")

    def _cluster(self, rel_path: str) -> str:
        """Return the directory portion of rel_path."""
        parent = str(Path(rel_path).parent)
        return parent if parent != "." else ""

    # ------------------------------------------------------------------
    # Layer 1 scan for a single file
    # ------------------------------------------------------------------

    def _scan_file(self, db: SemanticDB, file_path: Path, git_sha: Optional[str]) -> Dict[str, Any]:
        """Scan one file through Layer 1. Returns {file_id, symbol_count, symbols}."""
        rel = self._relative(file_path)
        try:
            stat = file_path.stat()
            last_modified = datetime.utcfromtimestamp(stat.st_mtime).isoformat()
            line_count = sum(1 for _ in file_path.open(encoding="utf-8", errors="replace"))
        except OSError:
            last_modified = ""
            line_count = 0

        file_id = db.upsert_file(
            path=rel,
            module_name=self._module_name(rel),
            cluster=self._cluster(rel),
            line_count=line_count,
            last_modified=last_modified,
            last_scan_sha=git_sha,
            scanned_at=datetime.utcnow().isoformat(),
        )

        # Clear stale data before re-inserting
        db.clear_file_data(file_id)

        raw_symbols = extract_symbols(file_path)
        raw_imports = extract_imports(file_path)

        # Insert imports
        for imp in raw_imports:
            db.insert_import(
                file_id,
                imp["imported_module"],
                imp.get("imported_name"),
                bool(imp["is_from_import"]),
            )

        # Insert symbols and call sites; annotate with db id for Layer 3
        inserted_symbols: List[Dict[str, Any]] = []
        for sym in raw_symbols:
            sym_id = db.insert_symbol(
                file_id=file_id,
                name=sym["name"],
                kind=sym["kind"],
                line_start=sym["line_start"],
                line_end=sym["line_end"],
                signature=sym.get("signature", ""),
                docstring=sym.get("docstring"),
                decorators=",".join(sym.get("decorators", [])),
            )
            # Extract call sites
            calls = extract_call_sites(file_path, sym)
            for call in calls:
                try:
                    db.insert_call_site(sym_id, call["callee_name"], call["line"])
                except (OSError, KeyError, TypeError):
                    pass

            sym_copy = dict(sym)
            sym_copy["_db_id"] = sym_id
            inserted_symbols.append(sym_copy)

        return {"file_id": file_id, "symbol_count": len(raw_symbols), "symbols": inserted_symbols}

    # ------------------------------------------------------------------
    # Layer 3 enrichment
    # ------------------------------------------------------------------

    def _run_layer3(self, db: SemanticDB, scan_info: List[Dict[str, Any]]) -> tuple[int, float]:
        """Run LLM enrichment. Returns (llm_calls, llm_cost_usd)."""
        if self.model_adapter is None:
            return 0, 0.0

        llm_calls = 0
        llm_cost = 0.0

        for info in scan_info:
            if llm_cost >= self.llm_budget:
                break
            file_id = info["file_id"]
            file_path = info.get("file_path")
            if file_path is None:
                continue

            # Module summary
            summary = generate_module_summary(db, file_id, file_path, self.model_adapter)
            if summary:
                llm_calls += 1

            # Function intent summaries (only for larger functions)
            eligible = [
                s for s in info.get("symbols", [])
                if s["kind"] in ("function", "method", "staticmethod", "classmethod")
                and (s["line_end"] - s["line_start"]) >= self.min_function_lines - 1
            ]
            if eligible:
                intents = generate_function_intents(
                    eligible, file_path, self.model_adapter, batch_size=self.batch_size
                )
                for sym_id, summary_text in intents.items():
                    if summary_text:
                        db.update_symbol_summary(sym_id, summary_text)
                # Count batches as calls
                import math
                llm_calls += math.ceil(len(eligible) / self.batch_size)

        return llm_calls, llm_cost

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_full(self, git_sha: str = "unknown") -> Dict[str, Any]:
        """Full scan: find .py files -> Layer 1 -> Layer 2 -> Layer 3 -> record_scan -> checkpoint.

        Returns {files_scanned, symbols_found, llm_calls, llm_cost_usd}
        """
        db = SemanticDB(self.db_path)
        try:
            py_files = self._find_py_files()
            rel_paths = [self._relative(f) for f in py_files]

            # Remove DB records for files that no longer exist on disk
            db.delete_missing_paths(rel_paths)

            total_symbols = 0
            scan_info: List[Dict[str, Any]] = []

            for fpath in py_files:
                info = self._scan_file(db, fpath, git_sha)
                info["file_path"] = fpath
                scan_info.append(info)
                total_symbols += info["symbol_count"]

            # Layer 2
            build_relationships(db)
            compute_coupling_scores(db)

            # Layer 3
            llm_calls, llm_cost = self._run_layer3(db, scan_info)

            # Record scan
            db.record_scan(
                scan_sha=git_sha,
                files_scanned=len(py_files),
                symbols_found=total_symbols,
                llm_calls_made=llm_calls,
                llm_cost_usd=llm_cost,
                scan_type="full",
                scan_time=datetime.utcnow().isoformat(),
            )
            db.checkpoint()

            return {
                "files_scanned": len(py_files),
                "symbols_found": total_symbols,
                "llm_calls": llm_calls,
                "llm_cost_usd": llm_cost,
            }
        finally:
            db.close()

    def scan_incremental(
        self, changed_files: List[str], git_sha: str = "unknown"
    ) -> Dict[str, Any]:
        """Re-scan only the listed files. Delete removed files. Rebuild relationships.

        changed_files: list of relative paths (e.g. ["core/foo.py"])
        Returns {files_scanned, symbols_found, llm_calls, llm_cost_usd}
        """
        db = SemanticDB(self.db_path)
        try:
            total_symbols = 0
            scan_info: List[Dict[str, Any]] = []

            for rel in changed_files:
                abs_path = self.project_root / rel
                if not abs_path.exists():
                    # File was deleted — remove from DB
                    existing = db.get_file_by_path(rel)
                    if existing:
                        db.delete_file(existing["id"])
                    continue

                info = self._scan_file(db, abs_path, git_sha)
                info["file_path"] = abs_path
                scan_info.append(info)
                total_symbols += info["symbol_count"]

            # Rebuild relationships and coupling for the whole DB
            build_relationships(db)
            compute_coupling_scores(db)

            # Layer 3
            llm_calls, llm_cost = self._run_layer3(db, scan_info)

            db.record_scan(
                scan_sha=git_sha,
                files_scanned=len(scan_info),
                symbols_found=total_symbols,
                llm_calls_made=llm_calls,
                llm_cost_usd=llm_cost,
                scan_type="incremental",
                scan_time=datetime.utcnow().isoformat(),
            )
            db.checkpoint()

            return {
                "files_scanned": len(scan_info),
                "symbols_found": total_symbols,
                "llm_calls": llm_calls,
                "llm_cost_usd": llm_cost,
            }
        finally:
            db.close()

    def refresh_if_needed(self) -> Optional[Dict[str, Any]]:
        """Check git HEAD vs last scan SHA. No-op if same.

        Falls back to full scan if:
        - No previous scan exists
        - git is unavailable
        - SHA is orphaned (differs from HEAD but no diff output)
        """
        db = SemanticDB(self.db_path)
        try:
            last = db.get_last_scan()
        finally:
            db.close()

        # Resolve current git HEAD
        try:
            current_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.project_root),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (OSError, subprocess.SubprocessError):
            # git unavailable — do a full scan
            return self.scan_full()

        if last is None:
            return self.scan_full(git_sha=current_sha)

        last_sha = last.get("scan_sha") or ""
        if last_sha == current_sha:
            # No change — no-op
            return None

        # Find changed files between last_sha and HEAD
        try:
            diff_output = subprocess.check_output(
                ["git", "diff", "--name-only", last_sha, current_sha],
                cwd=str(self.project_root),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (OSError, subprocess.SubprocessError):
            # SHA might be orphaned; do a full scan
            return self.scan_full(git_sha=current_sha)

        changed = [
            line for line in diff_output.splitlines()
            if line.endswith(".py")
        ]
        if not changed:
            # No Python files changed; record the new SHA without full re-scan
            return self.scan_incremental([], git_sha=current_sha)

        return self.scan_incremental(changed, git_sha=current_sha)
