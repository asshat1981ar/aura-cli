#!/usr/bin/env python3
"""Profile Python import times to identify bottlenecks.

Uses Python's built-in importtime tracing to analyze which imports
are taking the most time during startup.

Usage:
    python scripts/profile_imports.py
    python scripts/profile_imports.py --module aura_cli.cli_main
    python scripts/profile_imports.py --top 20
    python scripts/profile_imports.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImportEntry:
    """Single import entry from importtime output."""

    self_us: int
    cumulative_us: int
    depth: int
    package: str
    parent: str | None = None
    children: list["ImportEntry"] = field(default_factory=list)

    @property
    def self_ms(self) -> float:
        return self.self_us / 1000.0

    @property
    def cumulative_ms(self) -> float:
        return self.cumulative_us / 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "self_ms": round(self.self_ms, 3),
            "cumulative_ms": round(self.cumulative_ms, 3),
            "depth": self.depth,
        }


def parse_import_time(output: str) -> list[ImportEntry]:
    """Parse Python -X importtime output into structured entries."""
    entries: list[ImportEntry] = []

    # Pattern: import time: self [us] | cumulative | imported package
    pattern = re.compile(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+([\w._]+)")

    for line in output.strip().split("\n"):
        match = pattern.match(line.strip())
        if match:
            self_us = int(match.group(1))
            cumulative_us = int(match.group(2))
            package = match.group(3)

            # Calculate depth from line indentation (2 spaces per level)
            # Find the original line to count leading spaces
            depth = 0
            if line.startswith("import time:"):
                # Count dots in package name as proxy for depth
                depth = package.count(".")

            entries.append(
                ImportEntry(
                    self_us=self_us,
                    cumulative_us=cumulative_us,
                    depth=depth,
                    package=package,
                )
            )

    return entries


def run_import_time(module: str, cwd: Path) -> list[ImportEntry]:
    """Run Python with -X importtime and parse results."""
    cmd = [sys.executable, "-X", "importtime", "-c", f"import {module}"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
    )

    # importtime output goes to stderr
    output = result.stderr
    return parse_import_time(output)


def analyze_imports(entries: list[ImportEntry], top_n: int = 20) -> dict[str, Any]:
    """Analyze import entries and generate statistics."""
    if not entries:
        return {}

    # Sort by self time (import time excluding children)
    by_self_time = sorted(entries, key=lambda e: e.self_us, reverse=True)[:top_n]

    # Sort by cumulative time (total import time)
    by_cumulative = sorted(entries, key=lambda e: e.cumulative_us, reverse=True)[:top_n]

    # Group by top-level package
    package_times: dict[str, int] = {}
    for entry in entries:
        top_pkg = entry.package.split(".")[0]
        package_times[top_pkg] = package_times.get(top_pkg, 0) + entry.self_us

    top_packages = sorted(
        [(pkg, us) for pkg, us in package_times.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    return {
        "by_self_time": [e.to_dict() for e in by_self_time],
        "by_cumulative_time": [e.to_dict() for e in by_cumulative],
        "by_package": [{"package": pkg, "total_ms": round(us / 1000.0, 2)} for pkg, us in top_packages],
        "total_imports": len(entries),
        "total_self_us": sum(e.self_us for e in entries),
        "total_cumulative_us": max(e.cumulative_us for e in entries) if entries else 0,
    }


def print_analysis(analysis: dict[str, Any], json_output: bool = False) -> None:
    """Print analysis results."""
    if json_output:
        print(json.dumps(analysis, indent=2))
        return

    print("\n" + "=" * 80)
    print("IMPORT PROFILE ANALYSIS")
    print("=" * 80)

    total_self_ms = analysis.get("total_self_us", 0) / 1000.0
    total_cum_ms = analysis.get("total_cumulative_us", 0) / 1000.0

    print("\n📊 Summary:")
    print(f"   Total imports: {analysis.get('total_imports', 0)}")
    print(f"   Total self time: {total_self_ms:.2f} ms")
    print(f"   Total cumulative time: {total_cum_ms:.2f} ms")

    print("\n🔥 Top imports by self time (excluding children):")
    print("-" * 60)
    for entry in analysis.get("by_self_time", [])[:10]:
        print(f"   {entry['self_ms']:>8.2f} ms  {entry['package']}")

    print("\n📦 Top imports by cumulative time (including children):")
    print("-" * 60)
    for entry in analysis.get("by_cumulative_time", [])[:10]:
        print(f"   {entry['cumulative_ms']:>8.2f} ms  {entry['package']}")

    print("\n📁 Top packages by total time:")
    print("-" * 60)
    for pkg in analysis.get("by_package", [])[:10]:
        print(f"   {pkg['total_ms']:>8.2f} ms  {pkg['package']}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Find heavy imports
    heavy = [e for e in analysis.get("by_self_time", []) if e["self_ms"] > 50]
    if heavy:
        print("\n⚠️  Heavy imports detected (>50ms self time):")
        for entry in heavy[:5]:
            print(f"   - {entry['package']}: {entry['self_ms']:.2f}ms")
        print("\n   💡 Consider lazy loading these modules using core.lazy_imports")

    # Check for common heavy packages
    heavy_packages = [p for p in analysis.get("by_package", []) if p["total_ms"] > 100]
    if heavy_packages:
        print("\n🐌 Heavy packages detected (>100ms total):")
        for pkg in heavy_packages[:5]:
            print(f"   - {pkg['package']}: {pkg['total_ms']:.2f}ms")

    print()


def generate_optimization_report(modules: list[str], cwd: Path) -> dict[str, Any]:
    """Generate a comprehensive optimization report for multiple modules."""
    report: dict[str, Any] = {
        "modules": {},
        "overall_recommendations": [],
    }

    for module in modules:
        entries = run_import_time(module, cwd)
        analysis = analyze_imports(entries, top_n=30)
        report["modules"][module] = analysis

    # Aggregate findings
    all_heavy_imports: dict[str, float] = {}
    for module, analysis in report["modules"].items():
        for entry in analysis.get("by_self_time", []):
            pkg = entry["package"]
            time_ms = entry["self_ms"]
            if time_ms > 10:  # Only track significant imports
                all_heavy_imports[pkg] = max(all_heavy_imports.get(pkg, 0), time_ms)

    # Sort and add recommendations
    sorted_heavy = sorted(
        all_heavy_imports.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    report["heavy_imports_ranked"] = [{"package": pkg, "max_self_ms": round(ms, 2)} for pkg, ms in sorted_heavy[:20]]

    # Add recommendations
    recommendations = []
    for pkg, ms in sorted_heavy[:10]:
        if ms > 50:
            recommendations.append(f"Consider lazy loading '{pkg}' ({ms:.1f}ms) - use LazyImport from core.lazy_imports")
        elif ms > 20:
            recommendations.append(f"Optional: lazy load '{pkg}' ({ms:.1f}ms) if not always needed")

    report["overall_recommendations"] = recommendations

    return report


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Profile Python import times")
    parser.add_argument(
        "--module",
        "-m",
        type=str,
        default="aura_cli.cli_main",
        help="Module to profile (default: aura_cli.cli_main)",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default="",
        help="Comma-separated list of modules for comprehensive report",
    )
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=20,
        help="Number of top imports to show (default: 20)",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--report",
        "-r",
        action="store_true",
        help="Generate comprehensive optimization report",
    )

    args = parser.parse_args()
    project_root = Path(__file__).parent.parent

    if args.report or args.modules:
        # Comprehensive report
        modules = [m.strip() for m in args.modules.split(",") if m.strip()]
        if not modules:
            modules = ["aura_cli.cli_main", "core.orchestrator", "agents.registry"]

        print(f"Generating optimization report for: {', '.join(modules)}")
        report = generate_optimization_report(modules, project_root)

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("\n" + "=" * 80)
            print("COMPREHENSIVE IMPORT OPTIMIZATION REPORT")
            print("=" * 80)

            for module, analysis in report["modules"].items():
                total_ms = analysis.get("total_self_us", 0) / 1000.0
                print(f"\n📦 {module}: {total_ms:.2f}ms total")

            print("\n🔥 Heaviest imports across all modules:")
            print("-" * 60)
            for item in report.get("heavy_imports_ranked", [])[:15]:
                print(f"   {item['max_self_ms']:>8.2f} ms  {item['package']}")

            print("\n💡 Recommendations:")
            print("-" * 60)
            for rec in report.get("overall_recommendations", []):
                print(f"   • {rec}")
            print()
    else:
        # Single module analysis
        print(f"Profiling imports for: {args.module}")
        entries = run_import_time(args.module, project_root)
        analysis = analyze_imports(entries, top_n=args.top)
        print_analysis(analysis, args.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
