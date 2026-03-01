import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_main_subprocess(*argv: str):
    env = os.environ.copy()
    env.setdefault("AURA_SKIP_CHDIR", "1")
    return subprocess.run(
        [sys.executable, "main.py", *argv],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
