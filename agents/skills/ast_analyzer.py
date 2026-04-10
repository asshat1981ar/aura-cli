"""AST-aware code analysis skill for deep structural understanding.

Uses Python's built-in ast module to analyze:
- Function/class complexity (nesting depth, branch count)
- Code smells (too many arguments, deep nesting, long functions)
- Import dependency graph
- Dead code detection (unreachable after return/raise)
- Type annotation coverage
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from agents.skills.base import SkillBase, iter_py_files


@dataclass
class ASTMetrics:
    """Metrics collected from AST analysis."""

    file_path: str = ""
    functions: int = 0
    classes: int = 0
    max_nesting: int = 0
    avg_complexity: float = 0.0
    type_annotation_pct: float = 0.0
    smells: List[Dict[str, Any]] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    dead_code: List[Dict[str, Any]] = field(default_factory=list)


class ASTAnalyzerSkill(SkillBase):
    """Deep AST-based structural analysis of Python code."""

    name = "ast_analyzer"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        project_root = Path(input_data.get("project_root", "."))
        t0 = time.time()

        results: Dict[str, Any] = {
            "files_analyzed": 0,
            "total_functions": 0,
            "total_classes": 0,
            "smells": [],
            "complexity_hotspots": [],
            "type_coverage": 0.0,
            "import_graph": {},
            "dead_code": [],
        }

        py_files = list(iter_py_files(project_root))

        all_metrics: List[ASTMetrics] = []
        for py_file in py_files[:200]:  # Limit for performance
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source, filename=str(py_file))
                metrics = self._analyze_file(tree, str(py_file.relative_to(project_root)))
                all_metrics.append(metrics)
            except (SyntaxError, UnicodeDecodeError):
                continue

        # Aggregate results
        results["files_analyzed"] = len(all_metrics)
        results["total_functions"] = sum(m.functions for m in all_metrics)
        results["total_classes"] = sum(m.classes for m in all_metrics)

        # Collect smells and hotspots
        for m in all_metrics:
            results["smells"].extend(m.smells)
            if m.max_nesting > 4 or m.avg_complexity > 10:
                results["complexity_hotspots"].append(
                    {
                        "file": m.file_path,
                        "max_nesting": m.max_nesting,
                        "avg_complexity": m.avg_complexity,
                    }
                )
            results["dead_code"].extend(m.dead_code)
            if m.file_path and m.imports:
                results["import_graph"][m.file_path] = m.imports

        # Type coverage
        typed = sum(m.type_annotation_pct for m in all_metrics)
        results["type_coverage"] = round(typed / max(len(all_metrics), 1), 1)

        # Limit output size
        results["smells"] = results["smells"][:20]
        results["complexity_hotspots"] = sorted(
            results["complexity_hotspots"],
            key=lambda x: x["max_nesting"],
            reverse=True,
        )[:10]
        results["dead_code"] = results["dead_code"][:10]

        results["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
        return results

    def _analyze_file(self, tree: ast.AST, file_path: str) -> ASTMetrics:
        """Analyze a single file's AST."""
        metrics = ASTMetrics(file_path=file_path)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metrics.functions += 1
                self._check_function_smells(node, file_path, metrics)
            elif isinstance(node, ast.ClassDef):
                metrics.classes += 1
                self._check_class_smells(node, file_path, metrics)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._collect_imports(node, metrics)

        # Calculate nesting depth
        metrics.max_nesting = self._max_nesting_depth(tree)

        # Type annotation coverage
        metrics.type_annotation_pct = self._type_coverage(tree)

        return metrics

    def _check_function_smells(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        metrics: ASTMetrics,
    ) -> None:
        """Check for function-level code smells."""
        name = node.name
        # Too many arguments
        args = node.args
        arg_count = len(args.args) + len(args.posonlyargs) + len(args.kwonlyargs)
        if arg_count > 6:
            metrics.smells.append(
                {
                    "type": "too_many_args",
                    "file": file_path,
                    "function": name,
                    "line": node.lineno,
                    "count": arg_count,
                    "severity": "medium",
                }
            )

        # Long function (>50 lines)
        if hasattr(node, "end_lineno") and node.end_lineno:
            length = node.end_lineno - node.lineno
            if length > 50:
                metrics.smells.append(
                    {
                        "type": "long_function",
                        "file": file_path,
                        "function": name,
                        "line": node.lineno,
                        "length": length,
                        "severity": "low",
                    }
                )

        # Dead code after return
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, (ast.Return, ast.Raise)) and i < len(node.body) - 1:
                metrics.dead_code.append(
                    {
                        "file": file_path,
                        "function": name,
                        "line": stmt.lineno,
                        "type": "unreachable_after_return",
                    }
                )
                break

    def _check_class_smells(self, node: ast.ClassDef, file_path: str, metrics: ASTMetrics) -> None:
        """Check for class-level code smells."""
        method_count = sum(1 for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        if method_count > 20:
            metrics.smells.append(
                {
                    "type": "god_class",
                    "file": file_path,
                    "class": node.name,
                    "line": node.lineno,
                    "methods": method_count,
                    "severity": "high",
                }
            )

    def _collect_imports(self, node: ast.Import | ast.ImportFrom, metrics: ASTMetrics) -> None:
        """Collect import module names."""
        if isinstance(node, ast.ImportFrom) and node.module:
            metrics.imports.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                metrics.imports.append(alias.name)

    def _max_nesting_depth(self, tree: ast.AST, depth: int = 0) -> int:
        """Calculate the maximum nesting depth of control-flow structures."""
        max_depth = depth
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                child_depth = self._max_nesting_depth(node, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._max_nesting_depth(node, depth)
                max_depth = max(max_depth, child_depth)
        return max_depth

    def _type_coverage(self, tree: ast.AST) -> float:
        """Calculate percentage of functions with return type annotations."""
        total = 0
        annotated = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total += 1
                if node.returns is not None:
                    annotated += 1
        return (annotated / max(total, 1)) * 100
