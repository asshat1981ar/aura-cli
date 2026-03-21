"""Guaranteed-execution lifecycle hooks for AURA orchestrator phases.

Hooks are shell commands configured in aura.config.json that execute at
phase boundaries. They cannot be circumvented by the model — they run in
the host process, not in the LLM context.

Hook types:
- PrePhase: Runs before a phase. Can BLOCK (exit 2), MODIFY input (stdout JSON),
  or OBSERVE (exit 0).
- PostPhase: Runs after a phase. Can OBSERVE, LOG, or TRIGGER side effects.

Configuration in aura.config.json::

    {
        "hooks": {
            "pre_apply": [
                {"command": "python3 scripts/check_no_secrets.py", "blocking": true}
            ],
            "post_verify": [
                {"command": "python3 scripts/notify_slack.py", "blocking": false}
            ]
        }
    }
"""
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.logging_utils import log_json


class HookTiming(str, Enum):
    PRE = "pre"
    POST = "post"


class HookResult(str, Enum):
    PASS = "pass"
    BLOCK = "block"
    MODIFY = "modify"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class HookConfig:
    """Configuration for a single hook."""
    command: str
    blocking: bool = True
    timeout_seconds: int = 30
    description: str = ""
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data) -> "HookConfig":
        if isinstance(data, str):
            return cls(command=data)
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class HookExecution:
    """Record of a hook execution for audit trail."""
    hook: HookConfig
    timing: HookTiming
    phase: str
    result: HookResult
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)
    modified_input: dict | None = None


class HookEngine:
    """Executes lifecycle hooks with guaranteed execution semantics.

    Unlike prompt-based rules, hooks execute in the host process and
    cannot be circumvented by the LLM. They provide guaranteed security
    enforcement and workflow automation.
    """

    VALID_PHASES = [
        "ingest", "skill_dispatch", "plan", "critique", "synthesize",
        "act", "sandbox", "apply", "verify", "reflect",
    ]

    def __init__(self, config: dict | None = None):
        self.hooks: dict[str, list[HookConfig]] = {}
        self.history: list[HookExecution] = []
        if config:
            self._load_config(config)

    def _load_config(self, config: dict):
        hooks_config = config.get("hooks", {})
        for hook_key, hook_list in hooks_config.items():
            if not isinstance(hook_list, list):
                hook_list = [hook_list]
            self.hooks[hook_key] = [HookConfig.from_dict(h) for h in hook_list]

    def get_hooks(self, timing: HookTiming, phase: str) -> list[HookConfig]:
        key = f"{timing.value}_{phase}"
        return self.hooks.get(key, [])

    def run_pre_hooks(self, phase: str,
                      phase_input: dict) -> tuple[bool, dict]:
        """Run pre-phase hooks.

        Returns:
            (should_proceed, potentially_modified_input)
            If any blocking hook returns exit code 2, the phase is aborted.
        """
        hooks = self.get_hooks(HookTiming.PRE, phase)
        current_input = phase_input.copy()

        for hook in hooks:
            execution = self._execute_hook(hook, HookTiming.PRE, phase, current_input)
            self.history.append(execution)

            if execution.result == HookResult.BLOCK:
                log_json("WARN", "hook_blocked_phase",
                         details={"phase": phase, "command": hook.command})
                return False, current_input

            if execution.result == HookResult.MODIFY and execution.modified_input:
                current_input.update(execution.modified_input)

        return True, current_input

    def run_post_hooks(self, phase: str, phase_output: dict):
        """Run post-phase hooks. These are observational — they don't block."""
        hooks = self.get_hooks(HookTiming.POST, phase)
        for hook in hooks:
            execution = self._execute_hook(hook, HookTiming.POST, phase, phase_output)
            self.history.append(execution)

    def _execute_hook(self, hook: HookConfig, timing: HookTiming,
                      phase: str, context: dict) -> HookExecution:
        """Execute a single hook command."""
        env_vars = {
            "AURA_PHASE": phase,
            "AURA_HOOK_TIMING": timing.value,
            "AURA_CONTEXT": json.dumps(context, default=str)[:10000],
            **hook.env,
        }
        merged_env = {**os.environ, **env_vars}
        start = time.time()

        try:
            proc = subprocess.run(
                hook.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=hook.timeout_seconds,
                env=merged_env,
            )
            duration = time.time() - start

            result = HookResult.PASS
            modified_input = None

            if proc.returncode == 2 and hook.blocking:
                result = HookResult.BLOCK
            elif proc.returncode != 0:
                result = HookResult.ERROR
            elif proc.stdout.strip():
                try:
                    modified_input = json.loads(proc.stdout.strip())
                    result = HookResult.MODIFY
                except json.JSONDecodeError:
                    pass

            return HookExecution(
                hook=hook, timing=timing, phase=phase, result=result,
                stdout=proc.stdout[:5000], stderr=proc.stderr[:5000],
                exit_code=proc.returncode, duration_seconds=duration,
                modified_input=modified_input,
            )

        except subprocess.TimeoutExpired:
            return HookExecution(
                hook=hook, timing=timing, phase=phase,
                result=HookResult.TIMEOUT,
                duration_seconds=hook.timeout_seconds,
            )
        except Exception as exc:
            return HookExecution(
                hook=hook, timing=timing, phase=phase,
                result=HookResult.ERROR,
                stderr=str(exc),
                duration_seconds=time.time() - start,
            )

    def get_audit_log(self) -> list[dict]:
        """Get full audit trail of all hook executions."""
        return [
            {
                "phase": e.phase,
                "timing": e.timing.value,
                "command": e.hook.command,
                "result": e.result.value,
                "exit_code": e.exit_code,
                "duration": round(e.duration_seconds, 3),
                "timestamp": e.timestamp,
                "blocked": e.result == HookResult.BLOCK,
            }
            for e in self.history
        ]
