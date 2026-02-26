"""Skill: analyse test coverage using coverage.py or a heuristic fallback."""
from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json


def _run_cmd(cmd: List[str], cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=60)
        return result.stdout + result.stderr
    except Exception:
        return None


def _heuristic_coverage(root: Path) -> Dict[str, Any]:
    src_funcs: Dict[str, List[str]] = {}
    test_funcs: Dict[str, List[str]] = {}
    import ast as _ast
    for f in root.rglob("*.py"):
        if ".git" in f.parts or "node_modules" in f.parts or "__pycache__" in f.parts:
            continue
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
            tree = _ast.parse(src)
        except Exception:
            continue
        funcs = [n.name for n in _ast.walk(tree) if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))]
        rel = str(f.relative_to(root))
        if "test" in rel.lower():
            test_funcs[rel] = funcs
        else:
            src_funcs[rel] = funcs

    tested = set()
    for funcs in test_funcs.values():
        for fn in funcs:
            name = re.sub(r"^test_", "", fn)
            tested.add(name)

    missing = []
    for path, funcs in src_funcs.items():
        for fn in funcs:
            if fn not in tested and not fn.startswith("_"):
                missing.append({"file": path, "function": fn})

    total = sum(len(v) for v in src_funcs.values())
    covered = total - len(missing)
    pct = round(covered / max(total, 1) * 100, 1)
    return {"coverage_pct": pct, "missing_files": list(src_funcs.keys() - set(test_funcs.keys())), "untested_functions": missing[:50], "meets_target": pct >= 80, "method": "heuristic"}


class TestCoverageAnalyzerSkill(SkillBase):
    name = "test_coverage_analyzer"

    def run_incremental(self, project_root: str, changed_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run coverage analysis only on files changed since the last commit.

        If *changed_files* is ``None``, ``git diff --name-only HEAD~1 HEAD``
        is used to discover changed ``.py`` files.  Falls back to a full
        coverage run when git fails or no changed files are found.
        """
        root = Path(project_root)

        if changed_files is None:
            raw = _run_cmd(["git", "diff", "--name-only", "HEAD~1", "HEAD"], root)
            if not raw:
                return self._run({"project_root": project_root})
            changed_files = [f.strip() for f in raw.splitlines() if f.strip().endswith(".py")]

        if not changed_files:
            return self._run({"project_root": project_root})

        # Derive --cov=<module> args and find matching test files
        cov_args: List[str] = []
        test_files: List[str] = []
        for cf in changed_files:
            p = Path(cf)
            mod = str(p).replace("\\", "/").removesuffix(".py").replace("/", ".")
            cov_args.extend(["--cov", mod])
            for tf in root.rglob(f"test_{p.stem}.py"):
                rel = str(tf.relative_to(root))
                if rel not in test_files:
                    test_files.append(rel)

        if not test_files:
            # No matching test files found → full run
            return self._run({"project_root": project_root})

        cmd = ["python3", "-m", "pytest"] + cov_args + ["--cov-report=json", "-q"] + test_files
        _run_cmd(cmd, root)

        cov_json_path = root / "coverage.json"
        if cov_json_path.exists():
            try:
                data = json.loads(cov_json_path.read_text())
                files_data = {
                    f: fd for f, fd in data.get("files", {}).items()
                    if any(cf.replace("\\", "/") in f.replace("\\", "/") for cf in changed_files)
                }
                if files_data:
                    pct = round(
                        sum(fd.get("summary", {}).get("percent_covered", 0.0) for fd in files_data.values())
                        / len(files_data),
                        1,
                    )
                else:
                    pct = 0.0
                cov_json_path.unlink(missing_ok=True)
                log_json("INFO", "test_coverage_incremental_complete", details={"pct": pct, "changed_files": changed_files})
                return {
                    "coverage_pct": pct,
                    "changed_files": changed_files,
                    "files": files_data,
                    "missing_files": [],
                    "untested_functions": [],
                    "meets_target": pct >= 80,
                    "method": "incremental",
                }
            except Exception:
                pass

        return self._run({"project_root": project_root})

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if input_data.get("incremental"):
            return self.run_incremental(
                str(input_data.get("project_root", ".")),
                input_data.get("changed_files"),
            )

        project_root = Path(input_data.get("project_root", "."))
        min_target = int(input_data.get("min_target", 80))

        # Try running coverage.py
        _run_cmd(["python3", "-m", "coverage", "run", "-m", "pytest", "--tb=no", "-q", "--no-header"], project_root)
        cov_json_path = project_root / "coverage.json"
        _run_cmd(["python3", "-m", "coverage", "json", "-o", "coverage.json", "--quiet"], project_root)

        if cov_json_path.exists():
            try:
                data = json.loads(cov_json_path.read_text())
                summary = data.get("totals", {})
                pct = round(summary.get("percent_covered", 0.0), 1)
                missing_files = [f for f, fd in data.get("files", {}).items() if fd.get("summary", {}).get("percent_covered", 100) < min_target]
                cov_json_path.unlink(missing_ok=True)
                log_json("INFO", "test_coverage_analyzer_complete", details={"pct": pct, "method": "coverage.py"})
                return {"coverage_pct": pct, "missing_files": missing_files, "untested_functions": [], "meets_target": pct >= min_target, "method": "coverage.py"}
            except Exception:
                pass

        # Fallback
        result = _heuristic_coverage(project_root)
        result["meets_target"] = result["coverage_pct"] >= min_target
        log_json("INFO", "test_coverage_analyzer_complete", details={"pct": result["coverage_pct"], "method": "heuristic"})
        return result
