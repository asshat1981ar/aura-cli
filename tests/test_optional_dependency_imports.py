from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_blocked_import(module_name: str, *blocked_modules: str) -> subprocess.CompletedProcess[str]:
    blocked = ", ".join(repr(name) for name in blocked_modules)
    script = f"""
import builtins
import sys

sys.path.insert(0, {str(REPO_ROOT)!r})
blocked = {{{blocked}}}
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root in blocked:
        raise ImportError(f"blocked optional import for test: {{name}}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
__import__({module_name!r})
print("IMPORT_OK")
"""
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_model_adapter_import_succeeds_without_requests_or_numpy():
    result = run_blocked_import("core.model_adapter", "requests", "numpy")

    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout


def test_cli_main_import_succeeds_without_optional_runtime_dependencies():
    result = run_blocked_import(
        "aura_cli.cli_main",
        "requests",
        "numpy",
        "git",
    )

    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout
