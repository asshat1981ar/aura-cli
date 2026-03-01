"""Fast quality metrics after each cycle — runs in < 500ms."""
from __future__ import annotations
import py_compile
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any


def run_quality_snapshot(
    project_root: Path,
    changed_files: Optional[List[str]] = None,
) -> Dict:
    """
    Fast quality metrics — DO NOT run full pytest (too slow).
    Returns dict with keys:
      - test_count: int  (grep def test_ in tests/ | wc -l)
      - syntax_errors: List[str]  (py_compile on changed_files if provided)
      - import_errors: List[str]  (try importing changed modules)
      - elapsed_ms: float

    Never raises — returns {"error": str} on unexpected failure.
    """
    t0 = time.monotonic()
    try:
        project_root = Path(project_root)

        # Count test functions
        test_count = 0
        tests_dir = project_root / "tests"
        if tests_dir.is_dir():
            try:
                result = subprocess.run(
                    ["grep", "-r", "def test_", str(tests_dir)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                test_count = len(result.stdout.splitlines())
            except Exception:
                test_count = 0

        # Syntax check changed files
        syntax_errors: List[str] = []
        if changed_files:
            for fpath in changed_files:
                p = project_root / fpath if not Path(fpath).is_absolute() else Path(fpath)
                if not str(p).endswith(".py") or not p.exists():
                    continue
                try:
                    py_compile.compile(str(p), doraise=True)
                except py_compile.PyCompileError as e:
                    syntax_errors.append(str(e))

        # Import check changed modules
        import_errors: List[str] = []
        if changed_files:
            for fpath in changed_files:
                p = project_root / fpath if not Path(fpath).is_absolute() else Path(fpath)
                if not str(p).endswith(".py") or not p.exists():
                    continue
                # Convert path to module name
                try:
                    rel = p.relative_to(project_root)
                    module = str(rel).replace("/", ".").removesuffix(".py")
                    result = subprocess.run(
                        [sys.executable, "-c", f"import {module}"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(project_root),
                    )
                    if result.returncode != 0:
                        import_errors.append(f"{module}: {result.stderr.strip()}")
                except Exception as e:
                    import_errors.append(f"{fpath}: {e}")

        elapsed_ms = (time.monotonic() - t0) * 1000
        return {
            "test_count": test_count,
            "syntax_errors": syntax_errors,
            "import_errors": import_errors,
            "elapsed_ms": elapsed_ms,
        }
    except Exception as e:
        return {"error": str(e)}


def check_coverage_thresholds(
    project_root: Path,
    files: List[str],
    threshold: float = 80.0
) -> List[Dict[str, Any]]:
    """
    Checks if given files meet the coverage threshold.
    Requires coverage.json to exist in project_root.
    """
    import json
    cov_path = project_root / "coverage.json"
    if not cov_path.exists():
        return []

    try:
        data = json.loads(cov_path.read_text())
        files_data = data.get("files", {})
        gaps = []
        
        for f in files:
            # Normalize path for comparison
            matched = None
            for cov_f, fd in files_data.items():
                # Check if f is a suffix of cov_f or vice versa
                if f.replace("\\", "/") in cov_f.replace("\\", "/"):
                    matched = fd
                    break
            
            if matched:
                pct = matched.get("summary", {}).get("percent_covered", 0.0)
                if pct < threshold:
                    gaps.append({"file": f, "coverage": pct})
            else:
                # If file not in coverage data at all, it's 0%
                gaps.append({"file": f, "coverage": 0.0})
                
        return gaps
    except Exception:
        return []
