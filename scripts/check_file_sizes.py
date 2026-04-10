#!/usr/bin/env python3
"""Sprint 0 CI gate: fail if any Python file exceeds the per-file line-count limit.

Usage (standalone):
    python3 scripts/check_file_sizes.py

Exit codes:
    0  – all files within limit
    1  – one or more files exceed the limit (names and line counts are printed)

The allow-list below records files that were already over the limit when this
gate was introduced.  Each entry MUST include a TODO note with a target sprint
for breaking the file up.  New files that exceed the limit will *not* be added
to the allow-list automatically — a reviewer must consciously accept the debt.
"""

import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

LINE_LIMIT = 500

# Source roots to scan (relative to repo root)
SCAN_ROOTS = [
    "aura_cli",
    "core",
    "agents",
    "tools",
]

# ──────────────────────────────────────────────────────────────────────────────
# Known violators — files that exceeded the limit when this gate was introduced.
# TODO(s0-file-size-lint): break each file up by Sprint 1 or Sprint 2 at the
#   latest, then remove it from this allow-list.
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_LARGE_FILES: set[str] = {
    # (lines at gate introduction — 2025)
    "aura_cli/api_server.py",            # 2040 lines – TODO: split into routers/
    "core/orchestrator.py",              # 1990 lines – TODO: extract phase runners
    "aura_cli/dispatch.py",              # 1567 lines – TODO: split by command group
    "aura_cli/commands.py",              # 1363 lines – TODO: split by command group
    "core/model_adapter.py",             # 1131 lines – TODO: split provider adapters
    "aura_cli/cli_options.py",           # 1034 lines – TODO: split option groups
    "core/config_manager.py",            #  974 lines – TODO: extract validators
    "aura_cli/options.py",               #  957 lines – TODO: merge with cli_options
    "core/workflow_engine.py",           #  952 lines – TODO: extract step runners
    "core/async_orchestrator.py",        #  903 lines – TODO: extract phase handlers
    "core/context_graph.py",             #  839 lines – TODO: extract graph algorithms
    "tools/github_copilot_mcp.py",       #  830 lines – TODO: split MCP handlers
    "aura_cli/server.py",                #  802 lines – TODO: extract route modules
    "tools/coverage_gap_analyzer.py",    #  781 lines – TODO: split analysis/report
    "tools/mcp_server.py",               #  776 lines – TODO: split MCP handlers
    "core/agent_sdk/semantic_scanner.py",#  763 lines – TODO: extract scanners
    "core/agent_sdk/tool_registry.py",   #  752 lines – TODO: split registry/loader
    "core/evolution_loop.py",            #  750 lines – TODO: extract strategy runners
    "agents/brainstorming_bots.py",      #  748 lines – TODO: split per-bot module
    "agents/multi_agent_workflow.py",    #  724 lines – TODO: extract workflow steps
    "tools/auto_test_generator.py",      #  676 lines – TODO: split generators
    "core/dpop.py",                      #  656 lines – TODO: extract token builders
    "agents/adversarial/strategies.py",  #  652 lines – TODO: split strategy classes
    "core/capability_manager.py",        #  641 lines – TODO: extract loaders
    "agents/registry.py",               #  633 lines – TODO: split registry/loader
    "aura_cli/mcp_cli.py",              #  630 lines – TODO: split command modules
    "tools/agentic_loop_mcp.py",         #  618 lines – TODO: split MCP handlers
    "core/voting/engine.py",             #  583 lines – TODO: extract vote strategies
    "tools/sequential_thinking_mcp.py",  #  546 lines – TODO: split MCP handlers
    "core/improvement_loop.py",          #  525 lines – TODO: extract loop phases
    "core/agentic_evaluation.py",        #  521 lines – TODO: split evaluators
    "tools/aura_control_mcp.py",         #  510 lines – TODO: split control handlers
}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """Scan Python files and report any that exceed LINE_LIMIT."""
    repo_root = Path(__file__).parent.parent.resolve()
    offenders: list[tuple[str, int]] = []

    for root_name in SCAN_ROOTS:
        root_path = repo_root / root_name
        if not root_path.exists():
            continue
        for py_file in sorted(root_path.rglob("*.py")):
            rel = py_file.relative_to(repo_root).as_posix()
            line_count = sum(1 for _ in py_file.open(encoding="utf-8", errors="replace"))
            if line_count > LINE_LIMIT and rel not in KNOWN_LARGE_FILES:
                offenders.append((rel, line_count))

    if offenders:
        print(
            f"\n❌  File-size gate FAILED — the following files exceed {LINE_LIMIT} lines"
            " and are not in the known-violators allow-list:\n"
        )
        for path, count in sorted(offenders, key=lambda x: -x[1]):
            print(f"  {count:>5} lines  {path}")
        print(
            "\nOptions:\n"
            "  1. Refactor the file so it fits within the limit.\n"
            "  2. If this is unavoidable technical debt, add the file to\n"
            "     KNOWN_LARGE_FILES in scripts/check_file_sizes.py with a TODO note.\n"
        )
        return 1

    print(f"✅  File-size gate passed — all new files are within {LINE_LIMIT} lines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
