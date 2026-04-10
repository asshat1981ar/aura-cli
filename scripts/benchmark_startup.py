#!/usr/bin/env python3
"""Benchmark AURA CLI startup performance.

Measures cold start times for various CLI commands over multiple iterations.
Produces statistics and recommendations for optimization targets.

Usage:
    python scripts/benchmark_startup.py
    python scripts/benchmark_startup.py --iterations 10 --commands version,help
    python scripts/benchmark_startup.py --json  # Output results as JSON
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    command: str
    iterations: int
    times_ms: list[float] = field(default_factory=list)
    errors: int = 0
    
    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0
    
    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0
    
    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0
    
    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0.0
    
    @property
    def stdev_ms(self) -> float | None:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else None
    
    def to_dict(self) -> dict[str, Any]:
        result = {
            "command": self.command,
            "iterations": self.iterations,
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "errors": self.errors,
        }
        if self.stdev_ms is not None:
            result["stdev_ms"] = round(self.stdev_ms, 2)
        return result


# Performance targets (in milliseconds)
TARGETS = {
    "version": {"cold_start": 300, "import_core": 200},
    "help": {"cold_start": 400, "import_core": 300},
    "import": {"cold_start": 500, "import_core": 400},
}


def run_command(argv: list[str], cwd: Path) -> tuple[float, bool, str]:
    """Run a command and measure execution time.
    
    Returns:
        Tuple of (elapsed_time_ms, success, output)
    """
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, str(cwd / "main.py")] + argv,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
        )
        elapsed = (time.perf_counter() - start) * 1000
        success = result.returncode == 0
        output = result.stdout.strip() if success else result.stderr.strip()
        return elapsed, success, output
    except subprocess.TimeoutExpired:
        return 30000.0, False, "Timeout"
    except Exception as e:
        return 0.0, False, str(e)


def benchmark_command(
    command: str,
    iterations: int,
    project_root: Path,
) -> BenchmarkResult:
    """Benchmark a single command over multiple iterations."""
    result = BenchmarkResult(command=command, iterations=iterations)
    
    # Map command names to argv
    argv_map = {
        "version": ["--version"],
        "help": ["--help"],
        "json-help": ["--json-help"],
        "goal-status": ["goal", "status"],
        "mcp-tools": ["mcp", "tools"],
    }
    argv = argv_map.get(command, [f"--{command}"])
    
    for i in range(iterations):
        elapsed, success, output = run_command(argv, project_root)
        if success:
            result.times_ms.append(elapsed)
        else:
            result.errors += 1
            print(f"  Warning: Run {i+1} failed: {output[:100]}")
    
    return result


def benchmark_import_time(module: str, iterations: int) -> BenchmarkResult:
    """Benchmark import time for a module."""
    result = BenchmarkResult(command=f"import:{module}", iterations=iterations)
    
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            # Use subprocess to get cold import time
            proc = subprocess.run(
                [sys.executable, "-c", f"import {module}"],
                capture_output=True,
                timeout=30,
            )
            elapsed = (time.perf_counter() - start) * 1000
            if proc.returncode == 0:
                result.times_ms.append(elapsed)
            else:
                result.errors += 1
        except Exception:
            result.errors += 1
    
    return result


def print_results(results: list[BenchmarkResult], json_output: bool = False) -> None:
    """Print benchmark results."""
    if json_output:
        output = {
            "results": [r.to_dict() for r in results],
            "targets": TARGETS,
        }
        print(json.dumps(output, indent=2))
        return
    
    print("\n" + "=" * 80)
    print("AURA CLI STARTUP BENCHMARK RESULTS")
    print("=" * 80)
    
    for result in results:
        print(f"\n📊 Command: {result.command}")
        print("-" * 40)
        print(f"  Iterations: {result.iterations}")
        print(f"  Errors:     {result.errors}")
        print(f"  Min:        {result.min_ms:.1f} ms")
        print(f"  Max:        {result.max_ms:.1f} ms")
        print(f"  Mean:       {result.mean_ms:.1f} ms")
        print(f"  Median:     {result.median_ms:.1f} ms")
        if result.stdev_ms is not None:
            print(f"  Std Dev:    {result.stdev_ms:.1f} ms")
        
        # Check targets
        target = TARGETS.get(result.command, {}).get("cold_start")
        if target:
            status = "✅ PASS" if result.median_ms < target else "❌ FAIL"
            print(f"  Target:     {target} ms {status}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Generate recommendations
    slow_commands = [r for r in results if r.median_ms > 500]
    if slow_commands:
        print("\n🐌 Slow commands (>500ms):")
        for r in slow_commands:
            print(f"   - {r.command}: {r.median_ms:.1f}ms")
        print("   Consider using lazy imports for heavy dependencies.")
    
    failed_targets = []
    for r in results:
        target = TARGETS.get(r.command, {}).get("cold_start")
        if target and r.median_ms > target:
            failed_targets.append((r.command, r.median_ms, target))
    
    if failed_targets:
        print("\n⚠️  Failed targets:")
        for cmd, actual, target in failed_targets:
            print(f"   - {cmd}: {actual:.1f}ms (target: {target}ms)")
    else:
        print("\n✅ All targets met!")
    
    print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark AURA CLI startup performance"
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=5,
        help="Number of iterations per command (default: 5)",
    )
    parser.add_argument(
        "--commands",
        "-c",
        type=str,
        default="version,help,json-help",
        help="Comma-separated list of commands to benchmark",
    )
    parser.add_argument(
        "--imports",
        "-i",
        type=str,
        default="aura_cli,core,agents",
        help="Comma-separated list of modules to benchmark import time",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output results as JSON",
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    commands = [c.strip() for c in args.commands.split(",") if c.strip()]
    imports = [m.strip() for m in args.imports.split(",") if m.strip()]
    
    results: list[BenchmarkResult] = []
    
    # Benchmark commands
    print(f"Benchmarking {len(commands)} command(s) with {args.iterations} iteration(s)...")
    for cmd in commands:
        print(f"  Running: {cmd}...", end=" ", flush=True)
        result = benchmark_command(cmd, args.iterations, project_root)
        results.append(result)
        print(f"✓ ({result.median_ms:.1f}ms)")
    
    # Benchmark imports
    print(f"\nBenchmarking {len(imports)} module import(s)...")
    for module in imports:
        print(f"  Importing: {module}...", end=" ", flush=True)
        result = benchmark_import_time(module, args.iterations)
        results.append(result)
        print(f"✓ ({result.median_ms:.1f}ms)")
    
    # Print results
    print_results(results, args.json)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
