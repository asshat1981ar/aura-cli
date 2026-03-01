import importlib.util
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from aura_cli.cli_options import parse_cli_args
from tests.cli_entrypoint_test_utils import REPO_ROOT, run_main_subprocess as _run_main_subprocess
from tests.cli_snapshot_utils import normalized_json_text

SCRIPT_PATH = REPO_ROOT / "scripts" / "print_cli_contract_report.py"

def load_script_module():
    spec = importlib.util.spec_from_file_location("print_cli_contract_report", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def compact_json_text(raw_json: str) -> str:
    return json.dumps(json.loads(raw_json), sort_keys=True, separators=(",", ":")) + "\n"

def strip_aura_logs(text: str) -> str:
    """Removes JSON log lines (like config_loaded_from_file) from the text."""
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('{"ts":') and stripped.endswith('}'):
            try:
                json.loads(stripped)
                continue # Skip this log line
            except json.JSONDecodeError:
                pass
        clean_lines.append(line)
    
    result = "\n".join(clean_lines)
    if text.endswith("\n") and clean_lines:
        result += "\n"
    return result

def run_dispatch_with_report(report: dict, *argv: str):
    import aura_cli.cli_main as cli_main
    parsed = parse_cli_args(list(argv))
    out = io.StringIO()
    err = io.StringIO()
    with patch("aura_cli.contract_report.build_cli_contract_report", return_value=report):
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main.dispatch_command(parsed, project_root=REPO_ROOT, runtime_factory=None)
    return code, strip_aura_logs(out.getvalue()), strip_aura_logs(err.getvalue())

def run_script_main(*argv: str):
    mod = load_script_module()
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = mod.main(list(argv))
    return code, strip_aura_logs(out.getvalue()), strip_aura_logs(err.getvalue())

def run_script_main_with_report(report: dict, *argv: str):
    mod = load_script_module()
    out = io.StringIO()
    err = io.StringIO()
    with patch.object(mod, "build_cli_contract_report", return_value=report):
        with redirect_stdout(out), redirect_stderr(err):
            code = mod.main(list(argv))
    return code, strip_aura_logs(out.getvalue()), strip_aura_logs(err.getvalue())

def run_main_main(*argv: str):
    import main as main_module
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main_module.main(argv=list(argv))
    return code, strip_aura_logs(out.getvalue()), strip_aura_logs(err.getvalue())

def run_main_main_with_report(report: dict, *argv: str):
    import main as main_module
    out = io.StringIO()
    err = io.StringIO()
    with patch("aura_cli.contract_report.build_cli_contract_report", return_value=report):
        with redirect_stdout(out), redirect_stderr(err):
            code = main_module.main(argv=list(argv))
    return code, strip_aura_logs(out.getvalue()), strip_aura_logs(err.getvalue())

def run_script_subprocess(*argv: str):
    env = os.environ.copy()
    env.setdefault("AURA_SKIP_CHDIR", "1")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *argv],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    proc.stdout = strip_aura_logs(proc.stdout)
    proc.stderr = strip_aura_logs(proc.stderr)
    return proc

def run_main_subprocess(*argv: str):
    proc = _run_main_subprocess(*argv)
    proc.stdout = strip_aura_logs(proc.stdout)
    proc.stderr = strip_aura_logs(proc.stderr)
    return proc

def run_patched_main_subprocess(report: dict, *argv: str):
    env = os.environ.copy()
    env.setdefault("AURA_SKIP_CHDIR", "1")
    snippet = """
import json
import sys
from pathlib import Path
from unittest.mock import patch

repo_root = Path(sys.argv[1])
report = json.loads(sys.argv[2])
argv = sys.argv[3:]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import main as main_module

with patch("aura_cli.contract_report.build_cli_contract_report", return_value=report):
    raise SystemExit(main_module.main(argv=argv))
"""
    proc = subprocess.run(
        [sys.executable, "-c", snippet, str(REPO_ROOT), json.dumps(report), *argv],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    proc.stdout = strip_aura_logs(proc.stdout)
    proc.stderr = strip_aura_logs(proc.stderr)
    return proc
