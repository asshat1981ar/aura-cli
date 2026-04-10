#!/usr/bin/env python3
"""
scripts/find_coverage_gaps.py — Identify highest-impact modules to test next.

Combines coverage data with git change frequency and code complexity to produce
a prioritized list of modules needing test coverage.

Usage:
    python3 scripts/find_coverage_gaps.py [--coverage-xml coverage.xml] [--top 25]
"""
import argparse
import ast
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class ModuleRisk:
    path: str
    coverage_pct: float
    change_frequency: int
    complexity_score: float
    risk_score: float
    lines_of_code: int

    @property
    def priority_label(self) -> str:
        if self.risk_score > 70:
            return "🔴 CRITICAL"
        elif self.risk_score > 40:
            return "🟡 HIGH   "
        elif self.risk_score > 20:
            return "🟢 MEDIUM "
        return "⚪ LOW    "


def parse_coverage_xml(coverage_path: str) -> Dict[str, float]:
    coverage_map: Dict[str, float] = {}
    if not os.path.exists(coverage_path):
        print(f"Warning: {coverage_path} not found, using 0% for all modules")
        return coverage_map

    tree = ET.parse(coverage_path)
    root = tree.getroot()
    for cls in root.findall(".//class"):
        filename = cls.get("filename", "")
        line_rate = float(cls.get("line-rate", 0))
        coverage_map[filename] = round(line_rate * 100, 1)
    return coverage_map


def get_git_change_frequency(
    directories: List[str], since: str = "6 months ago"
) -> Dict[str, int]:
    freq_map: Dict[str, int] = {}
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for py_file in Path(directory).rglob("*.py"):
            try:
                result = subprocess.run(
                    ["git", "log", "--oneline", f"--since={since}", "--", str(py_file)],
                    capture_output=True, text=True, timeout=10,
                )
                freq_map[str(py_file)] = len(result.stdout.strip().splitlines())
            except (subprocess.TimeoutExpired, FileNotFoundError):
                freq_map[str(py_file)] = 0
    return freq_map


def calculate_complexity(filepath: str) -> float:
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
    except (SyntaxError, FileNotFoundError):
        return 0.0

    complexity = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity += 1

    try:
        with open(filepath) as f:
            loc = len([line for line in f.readlines() if line.strip()])
    except FileNotFoundError:
        loc = 1

    return round(complexity / max(loc / 100, 1), 2)


def count_lines(filepath: str) -> int:
    try:
        with open(filepath) as f:
            return len([line for line in f.readlines() if line.strip()])
    except FileNotFoundError:
        return 0


def analyze_modules(
    directories: List[str], coverage_map: Dict[str, float]
) -> List[ModuleRisk]:
    change_freq = get_git_change_frequency(directories)
    results = []

    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for py_file in Path(directory).rglob("*.py"):
            if py_file.name == "__init__.py" or "test" in py_file.name.lower():
                continue

            filepath = str(py_file)
            coverage = coverage_map.get(filepath, 0.0)
            changes = change_freq.get(filepath, 0)
            complexity = calculate_complexity(filepath)
            loc = count_lines(filepath)

            if loc < 10:
                continue

            coverage_risk = (100 - coverage) / 100
            change_weight = max(min(changes / 5, 3.0), 0.5)
            complexity_weight = max(min(complexity / 2, 3.0), 0.5)
            risk_score = round(coverage_risk * change_weight * complexity_weight * 100, 1)

            results.append(ModuleRisk(
                path=filepath,
                coverage_pct=coverage,
                change_frequency=changes,
                complexity_score=complexity,
                risk_score=risk_score,
                lines_of_code=loc,
            ))

    results.sort(key=lambda r: r.risk_score, reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Find coverage gaps by risk priority")
    parser.add_argument("--coverage-xml", default="coverage.xml", help="Path to coverage.xml")
    parser.add_argument("--top", type=int, default=25, help="Show top N modules (default: 25)")
    args = parser.parse_args()

    directories = ["aura_cli", "core", "agents", "memory"]
    print("🔍 AURA CLI — Coverage Gap Analysis\n")

    coverage_map = parse_coverage_xml(args.coverage_xml)
    results = analyze_modules(directories, coverage_map)

    if not results:
        print("No modules found to analyze.")
        return 0

    header = f"{'Priority':<14} {'Module':<50} {'Cov%':>5} {'Chg':>4} {'Cmplx':>6} {'LOC':>5} {'Risk':>6}"
    print(header)
    print("-" * len(header))

    for r in results[:args.top]:
        print(
            f"{r.priority_label:<14} {r.path:<50} {r.coverage_pct:>5.1f} "
            f"{r.change_frequency:>4} {r.complexity_score:>6.1f} "
            f"{r.lines_of_code:>5} {r.risk_score:>6.1f}"
        )

    total_loc = sum(r.lines_of_code for r in results)
    covered_loc = sum(r.lines_of_code * r.coverage_pct / 100 for r in results)
    critical = len([r for r in results if r.risk_score > 70])
    high = len([r for r in results if 40 < r.risk_score <= 70])

    print(f"\n{'='*60}")
    print(f"Total modules analyzed:    {len(results)}")
    print(f"Total LOC:                 {total_loc:,}")
    print(f"Estimated covered LOC:     {covered_loc:,.0f} ({covered_loc/max(total_loc,1)*100:.1f}%)")
    print(f"Critical priority modules: {critical}")
    print(f"High priority modules:     {high}")
    print(f"\n💡 Focus on the top {min(5, len(results))} modules for maximum impact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
