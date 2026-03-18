"""Skill that executes commands, captures output, and extracts diagnostics."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from agents.skills.base import SkillBase

@dataclass
class CommandResult:
    id: str
    exit_code: int
    duration_sec: float
    stdout: str
    stderr: str
    timed_out: bool = False

@dataclass
class DiagnosticLocation:
    file: str
    line: int
    col: int = 0

@dataclass
class DiagnosticSymbol:
    name: str
    confidence: float
    method: str

@dataclass
class Diagnostic:
    severity: str
    kind: str
    message: str
    primary_location: DiagnosticLocation
    symbol: DiagnosticSymbol | None = None
    stack: List[DiagnosticLocation] = field(default_factory=list)
    suggested_next_commands: List[List[str]] = field(default_factory=list)

def execute_command(run_config: dict) -> CommandResult:
    """
    Executes a command and captures its output.
    """
    start_time = time.time()
    process = None

    try:
        process = subprocess.Popen(
            run_config["cmd"],
            cwd=run_config.get("cwd", "."),
            env=run_config.get("env", os.environ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )

        stdout, stderr = process.communicate(timeout=run_config.get("timeout_sec", 600))

        return CommandResult(
            id=run_config["id"],
            exit_code=process.returncode,
            duration_sec=time.time() - start_time,
            stdout=stdout,
            stderr=stderr
        )

    except subprocess.TimeoutExpired as e:
        if process is not None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        return CommandResult(
            id=run_config["id"],
            exit_code=-1,
            duration_sec=time.time() - start_time,
            stdout=e.stdout or "",
            stderr=e.stderr or "",
            timed_out=True
        )
    except Exception as e:
        return CommandResult(
            id=run_config["id"],
            exit_code=-1,
            duration_sec=time.time() - start_time,
            stdout="",
            stderr=str(e)
        )

def parse_python_traceback(text: str) -> List[Diagnostic]:
    """Parse a Python traceback and return a diagnostic for the innermost frame."""
    diagnostics = []
    lines = text.strip().split('\n')
    if len(lines) >= 2 and any("Traceback" in line for line in lines):
        error_message = lines[-1]
        file_line_match = None
        for line in reversed(lines):
            file_line_match = re.search(r'File "(.*?)", line (\d+)', line)
            if file_line_match:
                break
        if file_line_match:
            file_path, lineno = file_line_match.groups()
            diagnostics.append(Diagnostic(
                severity="error",
                kind="python_traceback",
                message=error_message.strip(),
                primary_location=DiagnosticLocation(file=file_path, line=int(lineno)),
                suggested_next_commands=[
                    ["python3", "-m", "pytest", file_path, "-v", "--tb=long"],
                    ["python3", file_path],
                ],
            ))
    return diagnostics


def parse_pytest_output(text: str) -> List[Diagnostic]:
    """Parse pytest -v output and return a diagnostic per FAILED test."""
    diagnostics = []
    # Matches: FAILED path/to/test.py::Class::method - ExceptionType: message
    # or:      FAILED path/to/test.py::method
    pattern = re.compile(
        r"^FAILED\s+([\w/.\-]+\.py(?:::\w+)+)(?:\s+-\s+(.+))?$",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        node_id, message = match.group(1), match.group(2) or "test failed"
        # node_id is like "tests/test_foo.py::Class::method" — extract file part
        file_path = node_id.split("::")[0]
        diagnostics.append(Diagnostic(
            severity="error",
            kind="pytest_failure",
            message=message.strip(),
            primary_location=DiagnosticLocation(file=file_path, line=0),
            suggested_next_commands=[
                ["python3", "-m", "pytest", node_id, "-v", "--tb=short"],
            ],
        ))
    return diagnostics


def parse_node_stacktrace(text: str) -> List[Diagnostic]:
    """Parse a Node.js error stack and return a diagnostic for the innermost app frame."""
    diagnostics = []
    # Match the error type + message on a leading line before the stack frames
    err_match = re.search(r"^(\w*Error[^:\n]*|Error): (.+)$", text, re.MULTILINE)
    if not err_match:
        return diagnostics
    error_message = f"{err_match.group(1)}: {err_match.group(2).strip()}"

    # Find the innermost non-Node-internal frame: "at <anything> (<file>:<line>:<col>)"
    # Skip frames whose file starts with "node:" (internals) or "internal/"
    frame_pattern = re.compile(r"at .+? \((.+?):(\d+):(\d+)\)")
    for frame in frame_pattern.finditer(text):
        file_path, lineno, col = frame.groups()
        if file_path.startswith("node:") or file_path.startswith("internal/"):
            continue
        diagnostics.append(Diagnostic(
            severity="error",
            kind="node_stacktrace",
            message=error_message,
            primary_location=DiagnosticLocation(
                file=file_path, line=int(lineno), col=int(col)
            ),
            suggested_next_commands=[
                ["node", "--stack-trace-limit=20", file_path],
            ],
        ))
        break  # Only the innermost application frame
    return diagnostics


def parse_flake8_output(text: str) -> List[Diagnostic]:
    """Parse flake8 output and return a diagnostic per violation."""
    diagnostics = []
    # Matches: path/to/file.py:line:col: Exxxx message
    pattern = re.compile(
        r"^(.+\.py):(\d+):(\d+):\s+([EWCF]\d+)\s+(.+)$",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        file_path, lineno, col, code, message = match.groups()
        severity = "warning" if code.startswith("W") else "error"
        diagnostics.append(Diagnostic(
            severity=severity,
            kind="lint_violation",
            message=f"{code}: {message.strip()}",
            primary_location=DiagnosticLocation(
                file=file_path, line=int(lineno), col=int(col)
            ),
            suggested_next_commands=[
                ["flake8", "--select", code, file_path],
            ],
        ))
    return diagnostics


class TestAndObserveSkill(SkillBase):
    """Run commands, capture output, and extract actionable diagnostics."""

    name = "test_and_observe"
    __test__ = False

    def __init__(self, brain=None, model=None):
        super().__init__(brain=brain, model=model)
        self.parsers = {
            "python_traceback": parse_python_traceback,
            "pytest_failure": parse_pytest_output,
            "node_stacktrace": parse_node_stacktrace,
            "lint_violation": parse_flake8_output,
        }

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        runs = input_data.get("runs") or []
        results = [execute_command(run) for run in runs]

        diagnostics = []
        for result in results:
            if result.exit_code != 0:
                # Run all parsers over combined output — each uses distinct patterns
                combined = result.stdout + "\n" + result.stderr
                for parser in self.parsers.values():
                    diagnostics.extend(parser(combined))

        return {
            "status": "success" if all(r.exit_code == 0 for r in results) else "failure",
            "summary": {
                "runs_total": len(results),
                "runs_failed": sum(1 for r in results if r.exit_code != 0),
                "duration_sec": sum(r.duration_sec for r in results),
            },
            "runs": [asdict(r) for r in results],
            "diagnostics": [asdict(d) for d in diagnostics],
        }
