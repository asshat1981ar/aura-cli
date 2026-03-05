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
            preexec_fn=os.setsid
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
    """
    Parses a Python traceback and returns a list of diagnostics.
    """
    diagnostics = []
    lines = text.strip().split('\n')
    if len(lines) >= 2 and "Traceback" in lines[0]:
        error_message = lines[-1]
        file_line_match = None
        for line in reversed(lines):
            file_line_match = re.search(r'File "(.*?)", line (\d+)', line)
            if file_line_match:
                break
        if file_line_match:
            file, line = file_line_match.groups()
            diagnostics.append(Diagnostic(
                severity="error",
                kind="python_traceback",
                message=error_message.strip(),
                primary_location=DiagnosticLocation(file=file, line=int(line)),
            ))
    return diagnostics


class TestAndObserveSkill(SkillBase):
    """Run commands, capture output, and extract actionable diagnostics."""

    name = "test_and_observe"
    __test__ = False

    def __init__(self):
        super().__init__(brain=None, model=None)
        self.parsers = {
            "python_traceback": parse_python_traceback
        }

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        runs = input_data.get("runs") or []
        results = [execute_command(run) for run in runs]

        diagnostics = []
        for result in results:
            if result.exit_code != 0:
                for parser in self.parsers.values():
                    diagnostics.extend(parser(result.stderr))

        return {
            "status": "success" if all(r.exit_code == 0 for r in results) else "failure",
            "summary": {
                "runs_total": len(results),
                "runs_failed": sum(1 for r in results if r.exit_code != 0),
                "duration_sec": sum(r.duration_sec for r in results)
            },
            "runs": [asdict(r) for r in results],
            "diagnostics": [asdict(d) for d in diagnostics]
        }
