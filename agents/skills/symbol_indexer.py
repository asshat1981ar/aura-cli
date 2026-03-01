"""Skill: build a project-wide symbol index — classes, functions, methods with file/line/docstring."""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json


def _first_docstring(node: ast.AST) -> str:
    """Extract the first line of a docstring from a function/class body, if present."""
    body = getattr(node, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        doc = str(body[0].value.value)
        return doc.split("\n")[0].strip()[:120]
    return ""


def _get_imports(tree: ast.Module) -> List[str]:
    """Return list of module paths this file imports from."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            # For 'from a.b import c', we want 'a.b'
            imports.append(node.module)
    return list(dict.fromkeys(imports))  # deduplicated, order-preserving


def _index_file(source: str, rel_path: str, include_private: bool) -> List[Dict]:
    """Return symbol records for a single file."""
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return []

    symbols: List[Dict] = []

    def _should_include(name: str) -> bool:
        if include_private:
            return True
        return not name.startswith("_")

    # Walk top-level and class-level nodes
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if _should_include(node.name):
                symbols.append({
                    "name": node.name,
                    "qualified_name": node.name,
                    "type": "class",
                    "file": rel_path,
                    "line": node.lineno,
                    "docstring": _first_docstring(node),
                    "bases": [ast.unparse(b) if hasattr(ast, "unparse") else "" for b in node.bases],
                })
            # Methods inside class
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _should_include(item.name) or item.name in ("__init__", "__call__", "__str__", "__repr__"):
                        symbols.append({
                            "name": item.name,
                            "qualified_name": f"{node.name}.{item.name}",
                            "type": "method",
                            "file": rel_path,
                            "line": item.lineno,
                            "docstring": _first_docstring(item),
                            "parent_class": node.name,
                            "args": [a.arg for a in item.args.args if a.arg != "self"],
                        })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _should_include(node.name):
                symbols.append({
                    "name": node.name,
                    "qualified_name": node.name,
                    "type": "function",
                    "file": rel_path,
                    "line": node.lineno,
                    "docstring": _first_docstring(node),
                    "args": [a.arg for a in node.args.args],
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                })

    return symbols


class SymbolIndexerSkill(SkillBase):
    name = "symbol_indexer"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        project_root_str: Optional[str] = input_data.get("project_root", ".")
        include_private: bool = bool(input_data.get("include_private", False))
        project_root = Path(project_root_str)

        all_symbols: List[Dict] = []
        import_graph: Dict[str, List[str]] = {}
        files_indexed = 0
        errors: List[str] = []

        _SKIP_PARTS = {".git", "__pycache__", "node_modules", ".pytest_cache"}

        for f in sorted(project_root.rglob("*.py")):
            if any(p in f.parts for p in _SKIP_PARTS):
                continue
            try:
                source = f.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                errors.append(str(exc))
                continue

            rel = str(f.relative_to(project_root))
            symbols = _index_file(source, rel, include_private)
            all_symbols.extend(symbols)

            try:
                tree = ast.parse(source, filename=rel)
                import_graph[rel] = _get_imports(tree)
            except SyntaxError:
                import_graph[rel] = []

            files_indexed += 1

        # Build by-type counts
        by_type: Dict[str, int] = defaultdict(int)
        for sym in all_symbols:
            by_type[sym["type"]] += 1

        # Build name → locations index for quick lookup
        name_index: Dict[str, List[Dict]] = defaultdict(list)
        for sym in all_symbols:
            name_index[sym["name"]].append({"file": sym["file"], "line": sym["line"], "type": sym["type"]})

        log_json("INFO", "symbol_indexer_complete", details={
            "symbols": len(all_symbols),
            "files": files_indexed,
            "classes": by_type.get("class", 0),
            "functions": by_type.get("function", 0),
            "methods": by_type.get("method", 0),
        })

        return {
            "symbols": all_symbols,
            "symbol_count": len(all_symbols),
            "files_indexed": files_indexed,
            "by_type": dict(by_type),
            "import_graph": import_graph,
            "name_index": dict(name_index),
            "errors": errors[:10],
        }
