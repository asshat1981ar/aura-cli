import sys
import os
import shutil
import sqlite3
import subprocess
import argparse
from pathlib import Path

from core.config_manager import DEFAULT_CONFIG
from core.git_tools import GitTools, GitRepoError
from core.runtime_auth import (
    _clean_secret,
    resolve_config_api_key,
    runtime_provider_status,
    runtime_provider_summary,
)
from core.capability_manager import capability_doctor_check


def _resolve_active_embedding_model_id(cfg: dict) -> str | None:
    semantic_memory = cfg.get("semantic_memory", DEFAULT_CONFIG["semantic_memory"]) or {}
    local_profiles = cfg.get("local_model_profiles", {}) or {}
    local_routing = cfg.get("local_model_routing", {}) or {}

    embedding_model = semantic_memory.get("embedding_model")
    if isinstance(embedding_model, str):
        if embedding_model.startswith("local_profile:"):
            profile_name = embedding_model.split(":", 1)[1]
            profile = local_profiles.get(profile_name, {}) or {}
            return profile.get("embedding_model") or profile.get("model")
        if embedding_model in {"local-tfidf-svd-50d", "local/tfidf-svd-50d"}:
            return "local-tfidf-svd-50d"
        if embedding_model.startswith("openai/"):
            return embedding_model.split("/", 1)[1]
        if embedding_model:
            return embedding_model

    routed_profile_name = local_routing.get("embedding")
    if isinstance(routed_profile_name, str) and routed_profile_name:
        profile = local_profiles.get(routed_profile_name, {}) or {}
        return profile.get("embedding_model") or profile.get("model")

    model_routing = cfg.get("model_routing", {}) or {}
    fallback = model_routing.get("embedding")
    if isinstance(fallback, str) and fallback:
        return fallback.split("/", 1)[1] if fallback.startswith("openai/") else fallback
    return None


def check_embedding_index(project_root: Path, cfg: dict) -> tuple[str, str]:
    brain_db_rel = cfg.get("brain_db_path", DEFAULT_CONFIG["brain_db_path"])
    brain_db = project_root / brain_db_rel
    if not brain_db.exists():
        return "WARN", f"{brain_db_rel} not found"

    active_model = _resolve_active_embedding_model_id(cfg)
    if not active_model:
        return "WARN", "Unable to resolve active embedding model from config."

    try:
        conn = sqlite3.connect(str(brain_db), check_same_thread=False)
        try:
            total_records_row = conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()
            total_records = int(total_records_row[0]) if total_records_row else 0
            if total_records == 0:
                return "PASS", f"No semantic records indexed yet for {active_model}."

            rows = conn.execute(
                "SELECT model_id, COUNT(*) FROM embeddings GROUP BY model_id ORDER BY model_id"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        return "WARN", f"Semantic index not initialized ({exc})."
    except Exception as exc:
        return "WARN", f"Unable to inspect semantic index ({exc})."

    if not rows:
        return "WARN", (
            f"Semantic records exist but no embeddings are stored for active model {active_model}. "
            "Run `python3 main.py memory reindex --json`."
        )

    model_counts = {str(model_id): int(count) for model_id, count in rows}
    active_count = model_counts.get(active_model, 0)
    if active_count > 0:
        return "PASS", f"Active model {active_model} has {active_count} stored embeddings."

    available = ", ".join(f"{model} ({count})" for model, count in sorted(model_counts.items()))
    return "WARN", (
        f"Active embedding model {active_model} has no stored embeddings. "
        f"Found: {available}. Run `python3 main.py memory reindex --json`."
    )

def check_python_version():
    """Checks if the Python version is 3.9 or higher."""
    if sys.version_info >= (3, 9):
        return "PASS", f"Python version: {sys.version.split(' ')[0]}"
    else:
        return "FAIL", f"Python version: {sys.version.split(' ')[0]} (Requires Python 3.9+)"

def check_dependencies():
    """Checks for required third-party dependencies."""
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "requests": "requests",
        "pydantic": "pydantic",
        "dotenv": "python-dotenv",
        "numpy": "numpy",
        "git": "gitpython",
        "rich": "rich",
        "textblob": "textblob",
        "networkx": "networkx"
    }
    missing = []
    import importlib.util
    for module, pkg_name in required.items():
        if importlib.util.find_spec(module) is None:
            missing.append(pkg_name)
    
    if not missing:
        return "PASS", "All required dependencies are installed."
    else:
        return "FAIL", f"Missing dependencies: {', '.join(missing)}"

def check_env_vars(openrouter_api_key_arg: str = None): # Add argument
    """Checks for required environment variables."""
    gemini_cli = _clean_secret(os.getenv("GEMINI_CLI_PATH"))
    gemini_ready = False
    if gemini_cli:
        path = Path(gemini_cli)
        gemini_ready = path.is_file() and os.access(path, os.X_OK)

    status = {
        "openai": bool(_clean_secret(os.getenv("OPENAI_API_KEY"))),
        "openrouter": bool(
            _clean_secret(openrouter_api_key_arg)
            or _clean_secret(os.getenv("OPENROUTER_API_KEY"))
            or _clean_secret(os.getenv("AURA_API_KEY"))
        ),
        "local_model": bool(_clean_secret(os.getenv("AURA_LOCAL_MODEL_COMMAND"))),
        "gemini_cli": gemini_ready,
        "chat_ready": False,
        "embedding_ready": bool(_clean_secret(os.getenv("OPENAI_API_KEY"))),
    }
    status["chat_ready"] = any(
        [
            status["openai"],
            status["openrouter"],
            status["local_model"],
            status["gemini_cli"],
        ]
    )
    overall = "PASS" if status["chat_ready"] else "WARN"
    return overall, runtime_provider_summary(status)

def check_sqlite_write_access(repo_root: Path):
    """Checks for SQLite write access in the repository directory."""
    test_db_path = repo_root / "test_write_access.db"
    try:
        conn = sqlite3.connect(test_db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
        conn.commit()
        conn.close()
        os.remove(test_db_path)
        return "PASS", f"SQLite write access in '{repo_root}': OK"
    except Exception as e:
        return "FAIL", f"SQLite write access in '{repo_root}': Failed ({e})"

def check_git_status(repo_root: Path):
    """Checks if Git is installed and the current directory is a Git repository."""
    # Check if git command is available
    if not shutil.which("git"):
        return "FAIL", "Git is not installed or not in PATH."

    # Check if it's a Git repository
    try:
        # GitTools will raise GitRepoError if not a repo
        GitTools(repo_path=str(repo_root))
        return "PASS", "Git is installed and repository is initialized."
    except GitRepoError as e:
        return "FAIL", f"Git repository check: Failed ({e})"
    except Exception as e:
        return "FAIL", f"Git check encountered an unexpected error: {e}"


def check_pytest_and_run_tests(repo_root: Path, run_tests: bool, openrouter_api_key_arg: str = None): # Add argument
    """
    Checks if pytest is available and optionally runs tests.
    Returns status and message.
    """
    pytest_path = shutil.which("pytest")
    if not pytest_path:
        return "WARN", "Pytest is not installed or not in PATH. Skipping test execution."

    if not run_tests:
        return "WARN", "Pytest is available, but tests were not run (use --run-tests)."

    # Find all test files in the repo_root
    test_files = [str(f) for f in repo_root.glob('**/test_*.py') if not 'env' in f.parts] # Exclude virtual environments

    if not test_files:
        return "WARN", "No test files (test_*.py) found in the repository."

    try:
        # Create a copy of the current environment
        env_for_pytest = os.environ.copy()
        # Prioritize the command-line argument for OPENROUTER_API_KEY
        if openrouter_api_key_arg:
            env_for_pytest["OPENROUTER_API_KEY"] = openrouter_api_key_arg
        
        # Explicitly pass test files to pytest
        pytest_command = [pytest_path, "-q", f"--rootdir={repo_root}"] + test_files

        result = subprocess.run(
            pytest_command,
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=False, # Do not raise exception for non-zero exit codes
            env=env_for_pytest # Pass the modified environment variables
        )
        
        output = result.stdout.strip()
        if "== no tests ran in" in output or "ERROR" in output or "FAIL" in output or result.returncode != 0:
            return "FAIL", f"Pytest tests failed or no tests ran. Output: {output}"
        else:
            return "PASS", f"Pytest tests passed. Output: {output}"

    except Exception as e:
        return "FAIL", f"Failed to run pytest: {e}"


def check_subsystem_probe(probe_factory=None):
    """Run the lightweight subsystem probe and summarize the result for doctor."""
    if probe_factory is None:
        from core.health_monitor import SystemHealthProbe
        probe_factory = SystemHealthProbe

    try:
        report = probe_factory().run_all()
    except Exception as exc:
        return "WARN", f"probe error: {exc}"

    if report.all_ok:
        names = ", ".join(c.name for c in report.checks)
        return "PASS", f"All checks OK ({names})"

    failed = [f"{c.name}: {c.error}" for c in report.failed]
    passing = [c.name for c in report.checks if c.ok]
    detail = f"Failed: {'; '.join(failed)}"
    if passing:
        detail += f" | OK: {', '.join(passing)}"
    return "WARN", detail


def main():
    parser = argparse.ArgumentParser(description="AURA Doctor - System Health Check.")
    parser.add_argument("--run-tests", action="store_true", help="Run pytest to check test suite status.")
    parser.add_argument("--openrouter-api-key", type=str, help="Specify OPENROUTER_API_KEY directly.")
    args = parser.parse_args()

    print("AURA Doctor - System Health Check\n")

    # Correctly resolve the repository root (parent of the directory containing this file)
    repo_root = Path(__file__).resolve().parent.parent

    results = []

    # Python Version
    status, msg = check_python_version()
    results.append(f"Python Version: {status} - {msg}")

    # Dependencies
    status, msg = check_dependencies()
    results.append(f"Dependencies: {status} - {msg}")

    # Environment Variables (pass the argument)
    status, msg = check_env_vars(args.openrouter_api_key)
    results.append(f"Environment Variables: {status} - {msg}")

    # SQLite Write Access
    status, msg = check_sqlite_write_access(repo_root)
    results.append(f"SQLite Write Access: {status} - {msg}")

    # Git Status
    status, msg = check_git_status(repo_root)
    results.append(f"Git Status: {status} - {msg}")

    # Pytest and Test Run (pass the argument)
    status, msg = check_pytest_and_run_tests(repo_root, args.run_tests, args.openrouter_api_key)
    results.append(f"Pytest Tests: {status} - {msg}")

    print("\n--- Checklist ---")
    for result in results:
        print(f"- {result}")
    print("-----------------\n")

    overall_status = "PASS"
    for result in results:
        if "FAIL" in result:
            overall_status = "FAIL"
            break
        elif "WARN" in result and overall_status == "PASS":
            overall_status = "WARN"
    
    print(f"Overall Health: {overall_status}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Doctor v2 — Rich-formatted comprehensive health check
# ---------------------------------------------------------------------------

def run_doctor_v2(
    project_root: Path = None,
    rich_output: bool = True,
    capability_check=capability_doctor_check,
) -> list:
    """
    Run all AURA health checks and return results as a list of dicts.

    Each dict has: {"check": str, "status": "PASS"|"WARN"|"FAIL", "detail": str}

    Args:
        project_root: Override project root (default: auto-detect)
        rich_output:  Print rich-formatted table if True and rich is available
    """
    import importlib.util
    import json
    import time

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    results = []

    def _add(check: str, status: str, detail: str):
        results.append({"check": check, "status": status, "detail": detail})

    # 1. Python version
    v = sys.version_info
    if v >= (3, 9):
        _add("Python version", "PASS", f"{v.major}.{v.minor}.{v.micro}")
    else:
        _add("Python version", "FAIL", f"{v.major}.{v.minor}.{v.micro} (need ≥3.9)")

    # 2. Config file
    from core.config_manager import config
    cfg = config.effective_config
    if config.config_file.exists():
        _add("Config file", "PASS", f"aura.config.json ({len(cfg)} keys)")
    else:
        _add("Config file", "WARN", "aura.config.json not found (using defaults)")

    # 3. Chat providers
    config_api_key = resolve_config_api_key(cfg.get("api_key"))
    provider_status = runtime_provider_status(config_api_key=config_api_key)
    _add(
        "API key",
        "PASS" if provider_status["chat_ready"] else "WARN",
        runtime_provider_summary(provider_status),
    )

    # 4. Embeddings
    active_embedding_model = _resolve_active_embedding_model_id(cfg)
    _add(
        "Embeddings",
        "PASS" if provider_status["embedding_ready"] else "WARN",
        (
            f"Configured embedding backend: {active_embedding_model}"
            if provider_status["embedding_ready"] and active_embedding_model
            else "OPENAI_API_KEY not set; semantic search will fall back"
        ),
    )
    embedding_index_status, embedding_index_detail = check_embedding_index(project_root, cfg)
    _add("Embedding index", embedding_index_status, embedding_index_detail)

    # 5. Brain DB
    brain_db_rel = cfg.get("brain_db_path", DEFAULT_CONFIG["brain_db_path"])
    brain_db = project_root / brain_db_rel
    if brain_db.exists():
        try:
            conn = sqlite3.connect(str(brain_db), check_same_thread=False)
            row = conn.execute("SELECT COUNT(*) FROM memory").fetchone()
            wal = conn.execute("PRAGMA journal_mode").fetchone()
            count = row[0] if row else 0
            mode = wal[0] if wal else "unknown"
            conn.close()
            status = "PASS" if mode == "wal" else "WARN"
            _add("Brain DB", status, f"{brain_db_rel} ({count:,} entries, journal={mode})")
        except Exception as e:
            _add("Brain DB", "FAIL", str(e))
    else:
        _add("Brain DB", "WARN", f"{brain_db_rel} not found")

    # 6. Goal queue
    goal_queue_rel = cfg.get("goal_queue_path", DEFAULT_CONFIG["goal_queue_path"])
    goal_queue_path = project_root / goal_queue_rel
    if goal_queue_path.exists():
        try:
            goals = json.loads(goal_queue_path.read_text())
            _add("Goal queue", "PASS", f"{goal_queue_rel} ({len(goals)} goals)")
        except Exception as e:
            _add("Goal queue", "FAIL", f"{goal_queue_rel}: {e}")
    else:
        _add("Goal queue", "WARN", f"{goal_queue_rel} not found")

    # 6b. Capability bootstrap
    capability_status, capability_detail = capability_check(project_root, config=cfg)
    _add("Capability bootstrap", capability_status, capability_detail)

    # 7. Dependencies
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "numpy": "numpy",
        "rich": "rich",
    }
    missing = [pkg for mod, pkg in required.items()
               if importlib.util.find_spec(mod) is None]
    if missing:
        _add("Dependencies", "WARN", f"Missing: {', '.join(missing)}")
    else:
        _add("Dependencies", "PASS", "All core packages present")

    # 8. rich available (for TUI)
    if importlib.util.find_spec("rich") is not None:
        _add("TUI (rich)", "PASS", f"rich available — run: aura watch")
    else:
        _add("TUI (rich)", "WARN", "pip install rich (for TUI dashboard)")

    # 9. Git repo
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=project_root, timeout=5
        )
        if result.returncode == 0:
            _add("Git repo", "PASS", f"HEAD={result.stdout.strip()}")
        else:
            _add("Git repo", "WARN", "Not a git repo or git unavailable")
    except Exception:
        _add("Git repo", "WARN", "git not available")

    # 10. OpenRouter reachable (quick DNS check only)
    try:
        import socket
        socket.setdefaulttimeout(2)
        socket.getaddrinfo("openrouter.ai", 443)
        _add("OpenRouter DNS", "PASS", "openrouter.ai resolves")
    except Exception:
        _add("OpenRouter DNS", "WARN", "openrouter.ai not reachable (offline?)")

    # 11. Subsystem health probe
    subsystem_status, subsystem_detail = check_subsystem_probe()
    _add("Subsystem probe", subsystem_status, subsystem_detail)

    # Print rich table if requested
    if rich_output:
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box

            console = Console()
            table = Table(title="AURA Doctor v2", box=box.ROUNDED, show_lines=False)
            table.add_column("Check", style="bold", width=18)
            table.add_column("Status", width=6)
            table.add_column("Detail")

            _icons = {"PASS": "[green]✓[/green]", "WARN": "[yellow]![/yellow]", "FAIL": "[red]✗[/red]"}
            for r in results:
                icon = _icons.get(r["status"], "?")
                table.add_row(r["check"], icon, r["detail"])

            console.print(table)
        except ImportError:
            for r in results:
                icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}.get(r["status"], "?")
                print(f"  [{icon}] {r['check']:<20} {r['detail']}")

    return results
