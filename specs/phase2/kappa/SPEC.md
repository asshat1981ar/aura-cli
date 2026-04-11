# KAPPA Specification: Command Recording & Replay

> **Agent**: KAPPA  
> **Feature**: Command Recording & Replay  
> **Priority**: P1 (Critical Path)  
> **Duration**: 2 days (Week 1: Thu-Fri)  
> **Depends On**: ALPHA (DI container from Phase 1)

---

## Overview

KAPPA provides command recording and replay functionality for AURA CLI. Users can record sequences of commands, save them as YAML workflows, and replay them later with variable interpolation.

## Goals

1. Record CLI commands as they execute
2. Save recordings as YAML workflows
3. Replay recordings with variable substitution
4. Support conditional steps and retries
5. Integrate with workflow engine foundation for OMICRON

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Recording & Replay Flow                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│   RECORDING MODE                                             │
│   ==============                                             │
│                                                               │
│   aura record start --name "deploy"                          │
│       │                                                      │
│       ▼                                                      │
│   ┌──────────────┐                                          │
│   │ Recorder     │  Intercepts commands                     │
│   │ Session      │                                          │
│   └──────┬───────┘                                          │
│          │                                                   │
│          ▼                                                   │
│   aura goal add "Build project"                              │
│       │                                                      │
│       ▼                                                      │
│   ┌──────────────┐                                          │
│   │ Capture:     │  - Command                               │
│   │   Command    │  - Timestamp                             │
│   │   Context    │  - Working directory                     │
│   │   Output     │  - Environment                           │
│   │   Result     │  - Exit code                             │
│   └──────┬───────┘                                          │
│          │                                                   │
│   aura record stop                                           │
│       │                                                      │
│       ▼                                                      │
│   ┌──────────────┐                                          │
│   │ Save as YAML │  Export to .aura/recordings/             │
│   └──────────────┘                                          │
│                                                               │
│   REPLAY MODE                                                │
│   ===========                                                │
│                                                               │
│   aura replay deploy --vars "env=production"                 │
│       │                                                      │
│       ▼                                                      │
│   ┌──────────────┐                                          │
│   │ Load YAML    │  Parse workflow                          │
│   └──────┬───────┘                                          │
│          │                                                   │
│          ▼                                                   │
│   ┌──────────────┐                                          │
│   │ For each     │  - Substitute variables                  │
│   │ step:        │  - Check conditions                      │
│   │   Execute    │  - Run command                           │
│   │   Validate   │  - Retry if needed                       │
│   └──────────────┘                                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```
aura/recording/
├── __init__.py              # Public API exports
├── recorder.py              # Recording session management
├── replay.py                # Replay engine
├── models.py                # Data models (Recording, Step)
├── storage.py               # YAML serialization
├── variables.py             # Variable interpolation
└── validation.py            # Step validation
```

## Interface Design

### Public API

```python
# aura/recording/__init__.py

from .recorder import Recorder, RecordingSession
from .replay import ReplayEngine
from .models import Recording, RecordingStep

__all__ = [
    "Recorder",
    "RecordingSession",
    "ReplayEngine",
    "Recording",
    "RecordingStep",
    "record_command",    # Decorator for auto-recording
]

# Decorator for recording functions automatically
def record_command(name: str | None = None):
    """Decorator to record function execution."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            recorder = Recorder()
            with recorder.session(name or func.__name__) as session:
                result = await func(*args, **kwargs)
                session.add_step(
                    command=f"{func.__name__}({', '.join(map(repr, args))})",
                    result=result,
                )
                return result
        return wrapper
    return decorator
```

### Data Models

```python
# aura/recording/models.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class RecordingStep:
    """A single step in a recording."""
    command: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    working_dir: str = ""
    env_vars: dict = field(default_factory=dict)
    exit_code: int = 0
    output: str = ""
    status: StepStatus = StepStatus.PENDING
    condition: Optional[str] = None  # e.g., "${env} == 'production'"
    retries: int = 0
    retry_delay: int = 5
    
    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "timestamp": self.timestamp.isoformat(),
            "working_dir": self.working_dir,
            "env_vars": self.env_vars,
            "exit_code": self.exit_code,
            "output": self.output,
            "status": self.status.value,
            "condition": self.condition,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RecordingStep":
        return cls(
            command=data["command"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            working_dir=data.get("working_dir", ""),
            env_vars=data.get("env_vars", {}),
            exit_code=data.get("exit_code", 0),
            output=data.get("output", ""),
            status=StepStatus(data.get("status", "pending")),
            condition=data.get("condition"),
            retries=data.get("retries", 0),
            retry_delay=data.get("retry_delay", 5),
        )

@dataclass
class Recording:
    """A complete recording session."""
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    steps: list[RecordingStep] = field(default_factory=list)
    variables: dict = field(default_factory=dict)  # Default variables
    
    def to_yaml(self) -> str:
        import yaml
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "variables": self.variables,
            "steps": [step.to_dict() for step in self.steps],
        }
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Recording":
        import yaml
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Recording":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            steps=[RecordingStep.from_dict(s) for s in data.get("steps", [])],
            variables=data.get("variables", {}),
        )
```

### Recorder

```python
# aura/recording/recorder.py

from contextlib import contextmanager
from typing import Optional
import os

class RecordingSession:
    """Active recording session."""
    
    def __init__(self, name: str, storage_path: str):
        self.recording = Recording(name=name)
        self.storage_path = storage_path
        self._active = False
    
    def __enter__(self):
        self._active = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._active = False
        self._save()
    
    def add_step(self, command: str, **kwargs):
        """Add a step to the recording."""
        if not self._active:
            raise RuntimeError("Recording session not active")
        
        step = RecordingStep(
            command=command,
            working_dir=os.getcwd(),
            env_vars=dict(os.environ),
            **kwargs
        )
        self.recording.steps.append(step)
    
    def _save(self):
        """Save recording to disk."""
        path = os.path.join(self.storage_path, f"{self.recording.name}.yaml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.recording.to_yaml())

class Recorder:
    """Main recorder interface."""
    
    DEFAULT_STORAGE = "~/.aura/recordings"
    
    def __init__(self, storage_path: str | None = None):
        self.storage_path = os.path.expanduser(
            storage_path or self.DEFAULT_STORAGE
        )
        self._current_session: Optional[RecordingSession] = None
    
    @contextmanager
    def session(self, name: str):
        """Start a recording session."""
        if self._current_session is not None:
            raise RuntimeError("Recording already in progress")
        
        self._current_session = RecordingSession(name, self.storage_path)
        try:
            with self._current_session as session:
                yield session
        finally:
            self._current_session = None
    
    def is_recording(self) -> bool:
        """Check if recording is active."""
        return self._current_session is not None
    
    def list_recordings(self) -> list[str]:
        """List all saved recordings."""
        if not os.path.exists(self.storage_path):
            return []
        return [f.replace(".yaml", "") 
                for f in os.listdir(self.storage_path) 
                if f.endswith(".yaml")]
    
    def load(self, name: str) -> Recording:
        """Load a recording by name."""
        path = os.path.join(self.storage_path, f"{name}.yaml")
        with open(path, "r") as f:
            return Recording.from_yaml(f.read())
```

### Replay Engine

```python
# aura/recording/replay.py

import subprocess
import asyncio
from typing import Callable

class ReplayEngine:
    """Engine for replaying recordings."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._progress_callback: Optional[Callable[[RecordingStep, StepStatus], None]] = None
    
    def on_progress(self, callback: Callable[[RecordingStep, StepStatus], None]):
        """Set progress callback."""
        self._progress_callback = callback
    
    async def replay(
        self,
        recording: Recording,
        variables: dict | None = None,
        stop_on_error: bool = True,
    ) -> ReplayResult:
        """
        Replay a recording.
        
        Args:
            recording: The recording to replay
            variables: Variable overrides
            stop_on_error: Whether to stop on first failure
            
        Returns:
            ReplayResult with status and step results
        """
        vars_dict = {**recording.variables, **(variables or {})}
        results = []
        
        for i, step in enumerate(recording.steps):
            # Check condition
            if step.condition and not self._evaluate_condition(step.condition, vars_dict):
                step.status = StepStatus.SKIPPED
                results.append(StepResult(step, skipped=True))
                continue
            
            # Substitute variables
            command = self._substitute_variables(step.command, vars_dict)
            
            # Execute with retries
            result = await self._execute_step(step, command, vars_dict)
            results.append(result)
            
            if not result.success and stop_on_error:
                break
        
        return ReplayResult(
            recording_name=recording.name,
            steps=results,
            success=all(r.success or r.skipped for r in results),
        )
    
    async def _execute_step(
        self,
        step: RecordingStep,
        command: str,
        variables: dict,
    ) -> StepResult:
        """Execute a single step with retries."""
        attempt = 0
        max_retries = step.retries
        
        while attempt <= max_retries:
            step.status = StepStatus.RUNNING
            self._notify_progress(step, step.status)
            
            if self.dry_run:
                print(f"[DRY RUN] Would execute: {command}")
                step.status = StepStatus.SUCCESS
                return StepResult(step, success=True)
            
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                
                if result.returncode == 0:
                    step.status = StepStatus.SUCCESS
                    step.output = result.stdout
                    self._notify_progress(step, step.status)
                    return StepResult(step, success=True, output=result.stdout)
                
                if attempt < max_retries:
                    attempt += 1
                    await asyncio.sleep(step.retry_delay)
                else:
                    step.status = StepStatus.FAILED
                    self._notify_progress(step, step.status)
                    return StepResult(step, success=False, error=result.stderr)
                    
            except Exception as e:
                if attempt < max_retries:
                    attempt += 1
                    await asyncio.sleep(step.retry_delay)
                else:
                    step.status = StepStatus.FAILED
                    return StepResult(step, success=False, error=str(e))
    
    def _substitute_variables(self, template: str, variables: dict) -> str:
        """Substitute ${var} placeholders."""
        import re
        
        def replace(match):
            var_name = match.group(1)
            if var_name in variables:
                return str(variables[var_name])
            # Environment variable fallback
            return os.environ.get(var_name, match.group(0))
        
        return re.sub(r'\$\{(\w+)\}', replace, template)
    
    def _evaluate_condition(self, condition: str, variables: dict) -> bool:
        """Evaluate a condition expression."""
        # Simple condition evaluation
        # e.g., "${env} == 'production'"
        try:
            # Substitute variables first
            expr = self._substitute_variables(condition, variables)
            # Safe evaluation (limited scope)
            return eval(expr, {"__builtins__": {}}, {})
        except:
            return False
    
    def _notify_progress(self, step: RecordingStep, status: StepStatus):
        if self._progress_callback:
            self._progress_callback(step, status)

@dataclass
class StepResult:
    step: RecordingStep
    success: bool = False
    skipped: bool = False
    output: str = ""
    error: str = ""

@dataclass
class ReplayResult:
    recording_name: str
    steps: list[StepResult]
    success: bool
```

## CLI Commands

```python
# aura_cli/commands.py additions

@app.command("record")
def record_command(
    action: str = typer.Argument(..., help="start, stop, list, or delete"),
    name: str = typer.Option(None, "--name", "-n", help="Recording name"),
):
    """Record CLI commands for later replay."""
    recorder = Recorder()
    
    if action == "start":
        if not name:
            name = typer.prompt("Recording name")
        # Start recording session
        typer.echo(f"Started recording: {name}")
        # Store in container for global access
        container.register("recording_session", recorder.session(name))
        
    elif action == "stop":
        # Stop current session
        typer.echo("Stopped recording")
        
    elif action == "list":
        recordings = recorder.list_recordings()
        for r in recordings:
            typer.echo(f"  - {r}")
            
    elif action == "delete":
        if not name:
            name = typer.prompt("Recording name to delete")
        # Delete recording
        typer.echo(f"Deleted recording: {name}")

@app.command("replay")
def replay_command(
    name: str = typer.Argument(..., help="Recording name"),
    vars: str = typer.Option(None, "--vars", "-v", help="Variables (key=value,key2=value2)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be executed"),
):
    """Replay a recorded command sequence."""
    recorder = Recorder()
    recording = recorder.load(name)
    
    # Parse variables
    variables = {}
    if vars:
        for pair in vars.split(","):
            key, value = pair.split("=")
            variables[key.strip()] = value.strip()
    
    engine = ReplayEngine(dry_run=dry_run)
    
    async def run():
        result = await engine.replay(recording, variables)
        if result.success:
            typer.echo(typer.style("✓ Replay successful", fg=typer.colors.GREEN))
        else:
            typer.echo(typer.style("✗ Replay failed", fg=typer.colors.RED))
    
    asyncio.run(run())
```

## YAML Format

```yaml
# ~/.aura/recordings/deploy.yaml
name: deploy
description: Deploy application to production
created_at: "2026-04-10T10:30:00"
variables:
  env: staging
  version: latest

steps:
  - command: echo "Deploying ${version} to ${env}"
    condition: "${env} != 'production' or ${CI} == 'true'"
    
  - command: git pull origin main
    retries: 2
    retry_delay: 5
    
  - command: docker build -t myapp:${version} .
    
  - command: docker push myapp:${version}
    condition: "${skip_push} != 'true'"
    
  - command: kubectl set image deployment/myapp myapp=myapp:${version}
    retries: 3
    working_dir: /opt/k8s
```

## Test Strategy

```python
# tests/unit/recording/test_recorder.py

class TestRecorder:
    def test_start_session_creates_recording(self):
        """Starting a session should create a recording object."""
        
    def test_add_step_captures_context(self):
        """Adding a step should capture cwd and env."""
        
    def test_stop_saves_to_yaml(self):
        """Stopping should save to YAML file."""
        
    def test_list_recordings(self):
        """Should list all saved recordings."""

# tests/unit/recording/test_replay.py

class TestReplayEngine:
    async def test_replay_executes_commands(self):
        """Replay should execute recorded commands."""
        
    async def test_variable_substitution(self):
        """Should substitute ${var} placeholders."""
        
    async def test_condition_evaluation(self):
        """Should skip steps where condition is false."""
        
    async def test_retry_on_failure(self):
        """Should retry failed commands up to retry limit."""
        
    async def test_dry_run_no_execution(self):
        """Dry run should not execute commands."""
```

## Integration with OMICRON

KAPPA provides the foundation for OMICRON's workflow engine:

```python
# OMICRON will extend KAPPA models
from aura.recording.models import Recording, RecordingStep

class Workflow(Recording):
    """Extended recording with DAG support."""
    dependencies: dict[str, list[str]]  # step -> depends_on
    
class WorkflowStep(RecordingStep):
    """Extended step with DAG and parallel execution."""
    parallel: bool = False
    depends_on: list[str] = field(default_factory=list)
```

## Success Criteria

- [ ] Recording session functional
- [ ] YAML export/import working
- [ ] Replay engine with variable substitution
- [ ] Condition evaluation working
- [ ] Retry logic implemented
- [ ] 20+ unit tests passing
- [ ] CLI commands integrated
- [ ] Foundation for OMICRON workflows

---

*Specification created: 2026-04-10*  
*Depends on: IOTA completion*
