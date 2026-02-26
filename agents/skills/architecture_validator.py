"""Skill: validate project architecture – circular imports, coupling, layer violations."""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.skills.base import SkillBase
from core.logging_utils import log_json

_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".tox", ".venv", "venv", "dist", "build"}


def _module_name(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        return str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")
    except ValueError:
        return path.stem


def _get_imports(source: str) -> List[str]:
    """Return top-level module names imported by source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    return imports


def _detect_cycles(graph: Dict[str, List[str]]) -> List[List[str]]:
    """DFS-based cycle detection. Returns each cycle as an ordered path list."""
    visited: Set[str] = set()
    rec_stack: Set[str] = set()
    seen_cycles: Set[Tuple] = set()
    cycles: List[List[str]] = []

    def dfs(node: str, path: List[str]) -> None:
        visited.add(node)
        rec_stack.add(node)
        for neighbour in graph.get(node, []):
            if neighbour not in graph:
                continue
            if neighbour not in visited:
                dfs(neighbour, path + [node])
            elif neighbour in rec_stack:
                idx = (path + [node]).index(neighbour) if neighbour in path else 0
                cycle = (path + [node])[idx:] + [neighbour]
                key = tuple(sorted(set(cycle)))
                if key not in seen_cycles:
                    seen_cycles.add(key)
                    cycles.append(cycle)
        rec_stack.discard(node)

    for node in list(graph.keys()):
        if node not in visited:
            dfs(node, [])
    return cycles


def _find_hubs(graph: Dict[str, List[str]], top_n: int = 5) -> List[Dict]:
    """Return modules with the most incoming edges (high fan-in = fragile hubs)."""
    in_degree: Dict[str, int] = defaultdict(int)
    for deps in graph.values():
        for dep in deps:
            in_degree[dep] += 1
    sorted_hubs = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)
    return [{"module": m, "dependents": c} for m, c in sorted_hubs[:top_n] if c > 1]


def _check_layer_violations(
    graph: Dict[str, List[str]],
    forbidden_patterns: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Check for forbidden import directions (e.g. 'core' must not import from 'api').
    forbidden_patterns: list of {"from": "api", "to": "core"} rules.
    """
    if not forbidden_patterns:
        return []
    violations = []
    for mod, deps in graph.items():
        for rule in forbidden_patterns:
            src_layer = rule.get("from", "")
            tgt_layer = rule.get("to", "")
            if src_layer in mod:
                for dep in deps:
                    if tgt_layer in dep:
                        violations.append({
                            "module": mod,
                            "imports": dep,
                            "rule": f"'{src_layer}' must not import from '{tgt_layer}'",
                        })
    return violations


def _analyse_project(
    root: Path,
    forbidden_patterns: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    py_files = [
        f for f in root.rglob("*.py")
        if not any(part in _SKIP_DIRS for part in f.parts)
    ]

    graph: Dict[str, List[str]] = {}
    local_mod_prefixes: Set[str] = set()

    for f in py_files:
        mod = _module_name(f, root)
        local_mod_prefixes.add(mod.split(".")[0])
        graph[mod] = []

    for f in py_files:
        mod = _module_name(f, root)
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        imports = _get_imports(src)
        # Keep only intra-project imports
        local_imports = [
            i for i in imports
            if i in local_mod_prefixes
            and any(
                _module_name(pf, root).startswith(i)
                for pf in py_files
            )
        ]
        graph[mod] = local_imports

    cycles = _detect_cycles(graph)
    total_edges = sum(len(v) for v in graph.values())
    coupling_score = round(total_edges / max(len(graph), 1), 2)
    hubs = _find_hubs(graph)
    layer_violations = _check_layer_violations(graph, forbidden_patterns)

    warnings: List[str] = []
    if cycles:
        warnings.append(f"{len(cycles)} circular import chain(s) — breaks Python import order.")
    if coupling_score > 5:
        warnings.append(f"High coupling ({coupling_score} avg deps/module) — consider splitting.")
    if hubs:
        top = hubs[0]
        warnings.append(f"Hub module '{top['module']}' has {top['dependents']} dependents — single point of failure.")
    if layer_violations:
        warnings.append(f"{len(layer_violations)} architecture layer violation(s) detected.")

    top_importers = sorted(graph.items(), key=lambda x: len(x[1]), reverse=True)[:10]

    return {
        "project_root": str(root),
        "module_count": len(graph),
        "total_import_edges": total_edges,
        "coupling_score": coupling_score,
        "circular_deps": [" → ".join(c) for c in cycles],
        "cycle_count": len(cycles),
        "hub_modules": hubs,
        "layer_violations": layer_violations,
        "warnings": warnings,
        "top_importers": {k: len(v) for k, v in top_importers},
    }


class ArchitectureValidatorSkill(SkillBase):
    """
    Validate project architecture: circular imports, coupling score, hub detection,
    and optional layer-violation rules (e.g. 'core must not import from api').

    Input:
      project_root        — path to scan (default '.')
      paths               — list of project roots for multi-project scan
      forbidden_patterns  — list of {"from": "layer_a", "to": "layer_b"} rules
    """

    name = "architecture_validator"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        project_root_str: Optional[str] = input_data.get("project_root")
        paths: Optional[List[str]] = input_data.get("paths")
        forbidden_patterns: Optional[List[Dict]] = input_data.get("forbidden_patterns")

        # Multi-project
        if paths:
            results = []
            for p in paths:
                root = Path(p)
                if not root.exists():
                    results.append({"project_root": p, "error": "Path not found"})
                    continue
                results.append(_analyse_project(root, forbidden_patterns))
            total_cycles = sum(r.get("cycle_count", 0) for r in results if "error" not in r)
            log_json("INFO", "architecture_validator_multi", details={"projects": len(paths), "total_cycles": total_cycles})
            return {"projects_scanned": len(paths), "total_cycles": total_cycles, "results": results}

        root = Path(project_root_str or ".")
        result = _analyse_project(root, forbidden_patterns)
        log_json("INFO", "architecture_validator_complete", details={
            "modules": result["module_count"],
            "cycles": result["cycle_count"],
            "coupling": result["coupling_score"],
        })
        return result
