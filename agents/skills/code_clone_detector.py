"""Skill: detect exact and near-duplicate code blocks across a project."""
from __future__ import annotations
import ast
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agents.skills.base import SkillBase
from core.logging_utils import log_json


def _normalize_ast(node: ast.AST) -> str:
    """Return a stable string representation of AST structure, ignoring names/literals."""
    parts = [type(node).__name__]
    for child in ast.iter_child_nodes(node):
        parts.append(_normalize_ast(child))
    return "(" + ",".join(parts) + ")"


def _extract_functions(source: str, file_path: str, min_lines: int) -> List[Tuple[str, str, int, int, str]]:
    """Returns list of (hash, normalized, file, start_line, func_name)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            fn_lines = end - node.lineno + 1
            if fn_lines < min_lines:
                continue
            norm = _normalize_ast(node)
            h = hashlib.md5(norm.encode()).hexdigest()
            funcs.append((h, norm, file_path, node.lineno, node.name))
    return funcs


def _jaccard(a: str, b: str) -> float:
    a_tokens = set(a.split(","))
    b_tokens = set(b.split(","))
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / max(union, 1)


class CodeCloneDetectorSkill(SkillBase):
    name = "code_clone_detector"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        project_root = Path(input_data.get("project_root", "."))
        min_lines = int(input_data.get("min_lines", 6))
        threshold = float(input_data.get("similarity_threshold", 0.8))

        all_funcs: List[Tuple] = []
        for f in project_root.rglob("*.py"):
            if ".git" in f.parts or "node_modules" in f.parts or "__pycache__" in f.parts:
                continue
            try:
                src = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(f.relative_to(project_root))
            all_funcs.extend(_extract_functions(src, rel, min_lines))

        # Exact clones: group by hash
        by_hash: Dict[str, List] = defaultdict(list)
        for h, norm, fpath, line, name in all_funcs:
            by_hash[h].append({"file": fpath, "function": name, "line": line})

        exact_clones = []
        for h, locations in by_hash.items():
            if len(locations) > 1:
                exact_clones.append({"locations": locations, "size_lines": min_lines, "hash": h})

        # Near-duplicates: pairwise Jaccard (limit to 200 for performance)
        near_dups = []
        funcs_sample = all_funcs[:200]
        for i in range(len(funcs_sample)):
            for j in range(i + 1, len(funcs_sample)):
                if funcs_sample[i][0] == funcs_sample[j][0]:
                    continue  # already exact clone
                sim = _jaccard(funcs_sample[i][1], funcs_sample[j][1])
                if sim >= threshold:
                    near_dups.append({"locations": [{"file": funcs_sample[i][2], "function": funcs_sample[i][4], "line": funcs_sample[i][3]}, {"file": funcs_sample[j][2], "function": funcs_sample[j][4], "line": funcs_sample[j][3]}], "similarity": round(sim, 2)})

        suggestions = []
        if exact_clones:
            suggestions.append(f"Extract {len(exact_clones)} duplicate function(s) into a shared utility module.")
        if near_dups:
            suggestions.append(f"Review {len(near_dups)} near-duplicate pair(s) for potential consolidation.")

        log_json("INFO", "code_clone_detector_complete", details={"exact": len(exact_clones), "near": len(near_dups)})
        return {"exact_clones": exact_clones[:50], "near_duplicates": near_dups[:50], "clone_count": len(exact_clones) + len(near_dups), "consolidation_suggestions": suggestions, "functions_analyzed": len(all_funcs)}
