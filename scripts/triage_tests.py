#!/usr/bin/env python3
"""
scripts/triage_tests.py — Categorize all test files as passing, failing, or hanging.

Usage:
    python3 scripts/triage_tests.py [--timeout 30] [--output reports/test-triage.json]

Produces a JSON report and text summaries for CI integration.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass
class TestFileResult:
    path: str
    status: str  # "pass", "fail", "timeout", "error", "skip"
    duration_seconds: float = 0.0
    tests_collected: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_errors: int = 0
    error_message: str = ""


@dataclass
class TriageReport:
    timestamp: str = ""
    total_files: int = 0
    passing: List[str] = field(default_factory=list)
    failing: List[str] = field(default_factory=list)
    timing_out: List[str] = field(default_factory=list)
    erroring: List[str] = field(default_factory=list)
    results: List[TestFileResult] = field(default_factory=list)


def discover_test_files(test_dir: str = "tests") -> List[Path]:
    """Find all test_*.py and *_test.py files."""
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"Error: {test_dir} directory not found", file=sys.stderr)
        sys.exit(1)
    files = sorted(list(test_path.rglob("test_*.py")) + list(test_path.rglob("*_test.py")))
    return list(dict.fromkeys(files))


def run_single_test_file(test_file: Path, timeout: int, env_override: dict) -> TestFileResult:
    """Run a single test file with timeout and capture results."""
    import re

    start = time.monotonic()
    result = TestFileResult(path=str(test_file))

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(test_file),
                "-v",
                "--no-cov",
                "--no-header",
                f"--timeout={timeout}",
                "--tb=line",
                "-q",
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 10,
            env={**os.environ, **env_override},
        )
        result.duration_seconds = round(time.monotonic() - start, 2)

        if proc.returncode == 0:
            result.status = "pass"
        elif proc.returncode == 5:
            result.status = "skip"
            result.error_message = "No tests collected"
        else:
            result.status = "fail"
            lines = (proc.stdout + proc.stderr).strip().splitlines()
            result.error_message = "\n".join(lines[-5:])

        for line in (proc.stdout + proc.stderr).splitlines():
            for match in re.finditer(r"(\d+) passed", line):
                result.tests_passed = int(match.group(1))
            for match in re.finditer(r"(\d+) failed", line):
                result.tests_failed = int(match.group(1))
            for match in re.finditer(r"(\d+) error", line):
                result.tests_errors = int(match.group(1))

    except subprocess.TimeoutExpired:
        result.duration_seconds = round(time.monotonic() - start, 2)
        result.status = "timeout"
        result.error_message = f"Exceeded {timeout}s timeout"
    except Exception as e:
        result.duration_seconds = round(time.monotonic() - start, 2)
        result.status = "error"
        result.error_message = str(e)

    result.tests_collected = result.tests_passed + result.tests_failed + result.tests_errors
    return result


def generate_report(results: List[TestFileResult]) -> TriageReport:
    report = TriageReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_files=len(results),
        results=results,
    )
    for r in results:
        if r.status == "pass":
            report.passing.append(r.path)
        elif r.status == "fail":
            report.failing.append(r.path)
        elif r.status == "timeout":
            report.timing_out.append(r.path)
        elif r.status == "error":
            report.erroring.append(r.path)
    return report


def print_summary(report: TriageReport) -> None:
    total = report.total_files
    print("\n" + "=" * 60)
    print("AURA CLI — Test Triage Report")
    print("=" * 60)
    print(f"Total test files: {total}")
    print(f"  ✅ Passing:     {len(report.passing)}")
    print(f"  ❌ Failing:     {len(report.failing)}")
    print(f"  ⏱️  Timing out: {len(report.timing_out)}")
    print(f"  💥 Erroring:   {len(report.erroring)}")
    print()

    if report.timing_out:
        print("⏱️  HANGING TESTS (need investigation):")
        for path in report.timing_out:
            print(f"   - {path}")
        print()

    if report.failing:
        print("❌ FAILING TESTS:")
        for path in report.failing:
            result = next(r for r in report.results if r.path == path)
            print(f"   - {path}")
            if result.error_message:
                for line in result.error_message.splitlines()[:2]:
                    print(f"     {line}")
        print()

    if report.passing:
        print("✅ SAFE TEST LIST (copy to CI):")
        print("python -m pytest \\")
        for i, path in enumerate(report.passing):
            suffix = " \\" if i < len(report.passing) - 1 else ""
            print(f"  {path}{suffix}")
        print("  -v --timeout=30 --no-cov")


def main():
    parser = argparse.ArgumentParser(description="Triage AURA CLI test files")
    parser.add_argument("--timeout", type=int, default=30, help="Per-file timeout (default: 30)")
    parser.add_argument("--test-dir", default="tests", help="Test directory (default: tests)")
    parser.add_argument("--output", default="reports/test-triage.json", help="Output JSON path")
    args = parser.parse_args()

    env_override = {"AURA_SKIP_CHDIR": "1"}
    test_files = discover_test_files(args.test_dir)
    print(f"Discovered {len(test_files)} test files in {args.test_dir}/")
    print(f"Timeout per file: {args.timeout}s\n")

    results = []
    for i, test_file in enumerate(test_files, 1):
        print(f"[{i}/{len(test_files)}] ⏳ {test_file} ...", end="", flush=True)
        result = run_single_test_file(test_file, args.timeout, env_override)
        status_map = {"pass": "✅", "fail": "❌", "timeout": "⏱️", "error": "💥", "skip": "⏭️"}
        icon = status_map.get(result.status, "?")
        print(f"\r[{i}/{len(test_files)}] {icon} {test_file} ({result.duration_seconds}s)")
        results.append(result)

    report = generate_report(results)
    print_summary(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\n📄 Report written to {output_path}")

    safe_list_path = output_path.parent / "safe-tests.txt"
    with open(safe_list_path, "w") as f:
        for path in report.passing:
            f.write(f"{path}\n")
    print(f"📄 Safe test list written to {safe_list_path}")

    if report.timing_out or report.erroring:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
