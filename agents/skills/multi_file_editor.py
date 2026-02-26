"""Skill #23 — MultiFileEditorSkill: dependency-ordered change planning across files."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Set

from agents.skills.base import SkillBase

_STOPWORDS: Set[str] = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "it", "all", "with", "from", "this", "that", "be", "by", "as",
    "are", "was", "were", "has", "have", "had", "do", "does", "did",
}

_SKIP_DIRS: Set[str] = {"__pycache__", ".git", "node_modules", "venv", ".venv", ".tox"}


def _keywords(goal: str) -> List[str]:
    tokens = re.split(r"[\s\W]+", goal.lower())
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


def _collect_py_files(root: Path, allowed: List[str] | None) -> List[Path]:
    results: List[Path] = []
    for p in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        results.append(p)
    if allowed:
        allowed_set = {str(Path(f)) for f in allowed}
        results = [p for p in results if str(p) in allowed_set or p.name in allowed_set]
    return results


def _assign_priority(rel_path: str, score: int, max_score: int) -> int:
    name = Path(rel_path).name.lower()
    if "test" in name or rel_path.startswith("tests"):
        return 3
    if score == max_score:
        return 1
    return 2


class MultiFileEditorSkill(SkillBase):
    name = "multi_file_editor"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        goal: str = input_data.get("goal", "")
        project_root = Path(input_data.get("project_root", "."))
        files_filter: List[str] | None = input_data.get("files")
        symbol_map: Dict[str, Any] | None = input_data.get("symbol_map")
        max_files: int = int(input_data.get("max_files", 20))

        warnings: List[str] = []

        if not goal:
            warnings.append("goal not provided — returning empty plan")
            return {
                "change_plan": [],
                "affected_count": 0,
                "warnings": warnings,
                "goal": goal,
            }

        keywords = _keywords(goal)
        if not keywords:
            warnings.append("no meaningful keywords extracted from goal")

        # Files with high-priority hits from symbol_map
        symbol_priority_files: Set[str] = set()
        symbol_caller_files: Set[str] = set()
        if symbol_map:
            for sym_name, sym_info in symbol_map.items():
                sym_lower = sym_name.lower()
                if any(kw in sym_lower for kw in keywords):
                    file_val = sym_info.get("file") if isinstance(sym_info, dict) else None
                    if file_val:
                        symbol_priority_files.add(file_val)
                    callers = sym_info.get("callers", []) if isinstance(sym_info, dict) else []
                    for c in callers:
                        cfile = c.get("file") if isinstance(c, dict) else c
                        if cfile:
                            symbol_caller_files.add(cfile)
        else:
            warnings.append("symbol_map not provided — using filename heuristics")

        # Collect candidate files
        try:
            py_files = _collect_py_files(project_root, files_filter)
        except Exception as exc:
            warnings.append(f"could not walk project_root ({exc}) — using empty file list")
            py_files = []

        # Score files
        scored: List[tuple[int, str, str]] = []  # (score, rel_path, abs_path)
        for p in py_files:
            try:
                rel = str(p.relative_to(project_root))
            except ValueError:
                rel = str(p)

            score = 0
            name_lower = p.name.lower()

            if rel in symbol_priority_files or p.name in symbol_priority_files:
                score += 3
            elif rel in symbol_caller_files or p.name in symbol_caller_files:
                score += 2

            if any(kw in name_lower for kw in keywords):
                score += 3

            if score == 0:
                score = 1

            scored.append((score, rel, str(p)))

        scored.sort(key=lambda x: -x[0])
        scored = scored[:max_files]

        max_score = scored[0][0] if scored else 1

        change_plan = []
        for score, rel, _abs in scored:
            priority = _assign_priority(rel, score, max_score)
            is_define = rel in symbol_priority_files or any(kw in Path(rel).name.lower() for kw in keywords)
            action = "rename definition" if is_define else "update references"
            change_plan.append({
                "file": rel,
                "reason": _reason(rel, score, symbol_priority_files, symbol_caller_files, keywords),
                "priority": priority,
                "suggested_action": action,
            })

        change_plan.sort(key=lambda x: (x["priority"], -scored[[r for _, r, _ in scored].index(x["file"])][0]
                                        if x["file"] in [r for _, r, _ in scored] else 0))

        return {
            "change_plan": change_plan,
            "affected_count": len(change_plan),
            "warnings": warnings,
            "goal": goal,
        }


def _reason(rel: str, score: int, sym_files: Set[str], caller_files: Set[str], keywords: List[str]) -> str:
    name = Path(rel).name.lower()
    parts = []
    if rel in sym_files:
        parts.append("defines a matching symbol")
    if rel in caller_files:
        parts.append("calls a matching symbol")
    if any(kw in name for kw in keywords):
        parts.append("filename matches goal keywords")
    if not parts:
        parts.append("candidate file in project")
    return "; ".join(parts)
