"""Skill: compute code diffs, detect symbol changes, measure churn, and analyse impact."""
from __future__ import annotations

import ast
import difflib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.skills.base import SkillBase
from core.logging_utils import log_json

_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------

def _get_symbols(source: str) -> Dict[str, int]:
    """Return {symbol_name: line_number} for all top-level defs and classes."""
    result: Dict[str, int] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return result
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = getattr(node, "lineno", 0)
    return result


def _symbols_modified(
    old_code: str,
    new_code: str,
    common: Set[str],
) -> List[str]:
    """
    Among symbols present in both old and new, detect which ones actually
    changed by comparing their extracted source segment lengths.
    """
    changed: List[str] = []
    try:
        old_tree = ast.parse(old_code)
        new_tree = ast.parse(new_code)
    except SyntaxError:
        return sorted(common)

    def _end_line(node: ast.AST) -> int:
        return getattr(node, "end_lineno", getattr(node, "lineno", 0))

    def _node_map(tree: ast.Module) -> Dict[str, Tuple[int, int]]:
        m: Dict[str, Tuple[int, int]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                m[node.name] = (getattr(node, "lineno", 0), _end_line(node))
        return m

    old_map = _node_map(old_tree)
    new_map = _node_map(new_tree)
    for name in common:
        old_span = old_map.get(name, (0, 0))
        new_span = new_map.get(name, (0, 0))
        old_len = old_span[1] - old_span[0]
        new_len = new_span[1] - new_span[0]
        if old_len != new_len:
            changed.append(name)
    return sorted(changed)


# ---------------------------------------------------------------------------
# Impact analysis
# ---------------------------------------------------------------------------

def _find_importers(symbols: List[str], project_root: Path, limit: int = 20) -> List[str]:
    """Find files that reference any of the given symbol names (text search)."""
    seen: Set[str] = set()
    results: List[str] = []
    for f in project_root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in f.parts):
            continue
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(sym in src for sym in symbols):
            rel = str(f.relative_to(project_root))
            if rel not in seen:
                seen.add(rel)
                results.append(rel)
                if len(results) >= limit:
                    break
    return results


# ---------------------------------------------------------------------------
# Git diff fallback
# ---------------------------------------------------------------------------

def _git_diff_files(project_root: Path) -> Optional[str]:
    """Return `git diff HEAD` output if git is available, else None."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=10,
        )
        return result.stdout if result.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Churn stats
# ---------------------------------------------------------------------------

def _churn(diff_lines: List[str]) -> Dict[str, int]:
    added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
    return {"lines_added": added, "lines_removed": removed, "churn": added + removed}


class IncrementalDifferSkill(SkillBase):
    """
    Compute unified diffs between old/new code versions, detect symbol changes
    (added / removed / modified), calculate churn metrics, identify impacted files,
    and optionally fall back to `git diff HEAD` when no code is supplied.

    Input:
      old_code      — previous version of the file (string)
      new_code      — updated version (string)
      file_path     — display label for the diff header (default 'unknown')
      project_root  — used for impact analysis of removed symbols
      use_git       — if True and no old/new code given, runs 'git diff HEAD'
    """

    name = "incremental_differ"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        old_code: str = input_data.get("old_code", "") or ""
        new_code: str = input_data.get("new_code", "") or ""
        file_path: str = input_data.get("file_path", "unknown")
        project_root_str: Optional[str] = input_data.get("project_root")
        use_git: bool = input_data.get("use_git", False)

        # Git fallback
        if not old_code and not new_code and use_git and project_root_str:
            git_out = _git_diff_files(Path(project_root_str))
            if git_out:
                return {
                    "diff_summary": git_out[:4000],
                    "source": "git_diff_head",
                    "added_symbols": [],
                    "removed_symbols": [],
                    "modified_symbols": [],
                    "impact_files": [],
                    "churn": {},
                    "rollback_plan": "Run `git checkout HEAD -- <file>` for each changed file.",
                    "diff_lines": git_out.count("\n"),
                }

        if not old_code and not new_code:
            return {"error": "Provide 'old_code' and/or 'new_code', or set use_git=true with project_root"}

        old_lines = old_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        ))
        diff_summary = "".join(diff[:80]) if diff else "No changes"
        churn = _churn(diff)

        old_sym_map = _get_symbols(old_code)
        new_sym_map = _get_symbols(new_code)
        old_names: Set[str] = set(old_sym_map)
        new_names: Set[str] = set(new_sym_map)

        added = sorted(new_names - old_names)
        removed = sorted(old_names - new_names)
        common = old_names & new_names
        modified = _symbols_modified(old_code, new_code, common)

        impact_files: List[str] = []
        if project_root_str and removed:
            impact_files = _find_importers(removed[:10], Path(project_root_str))

        rollback = (
            f"To rollback: restore '{file_path}' from the old_code snapshot."
            if old_code
            else "No rollback info — old_code was not provided."
        )

        log_json("INFO", "incremental_differ_complete", details={
            "added": len(added),
            "removed": len(removed),
            "modified": len(modified),
            "churn": churn["churn"],
        })

        return {
            "diff_summary": diff_summary,
            "diff_lines": len(diff),
            "churn": churn,
            "added_symbols": added,
            "removed_symbols": removed,
            "modified_symbols": modified,
            "impact_files": impact_files,
            "rollback_plan": rollback,
            "source": "inline",
        }
