import sys
import os
import shutil
import sqlite3
import subprocess
import argparse
from pathlib import Path

from core.git_tools import GitTools, GitRepoError

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
    status = "PASS"
    messages = []

    # Check for OPENROUTER_API_KEY, prioritizing command-line argument
    if openrouter_api_key_arg or os.getenv("OPENROUTER_API_KEY"):
        messages.append("OPENROUTER_API_KEY: Present")
    else:
        status = "WARN"
        messages.append("OPENROUTER_API_KEY: Not found (API functionality might be limited)")
    
    return status, "; ".join(messages)

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


def main():
    parser = argparse.ArgumentParser(description="AURA Doctor - System Health Check.")
    parser.add_argument("--run-tests", action="store_true", help="Run pytest to check test suite status.")
    parser.add_argument("--openrouter-api-key", type=str, help="Specify OPENROUTER_API_KEY directly.")
    args = parser.parse_args()

    print("AURA Doctor - System Health Check\n")

    repo_root = Path(__file__).resolve().parent

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

