"""Python bridge for invoking the Node-based BEADS adapter."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.beads_contract import (
    BEADS_SCHEMA_VERSION,
    BeadsBridgeConfig,
    BeadsDecision,
    BeadsInput,
    BeadsResult,
)
from core.beads_cli import resolve_beads_cli
from core.logging_utils import log_json
from core.operator_runtime import build_queue_summary


def default_beads_command(project_root: Path) -> tuple[str, ...]:
    return ("node", str(Path(project_root) / "scripts" / "beads_bridge.mjs"))


def build_beads_input(
    *,
    goal: str,
    project_root: Path,
    runtime_mode: str,
    queue_summary: Mapping[str, Any],
    active_context: Mapping[str, Any] | None = None,
    prd_context: Mapping[str, Any] | None = None,
    conductor_track: Mapping[str, Any] | None = None,
    goal_type: str | None = None,
) -> BeadsInput:
    return {
        "schema_version": BEADS_SCHEMA_VERSION,
        "goal": goal,
        "goal_type": goal_type,
        "runtime_mode": runtime_mode,
        "project_root": str(project_root),
        "queue_summary": dict(queue_summary),
        "active_context": dict(active_context or {}),
        "prd_context": dict(prd_context) if prd_context is not None else None,
        "conductor_track": dict(conductor_track) if conductor_track is not None else None,
    }


def load_prd_context(project_root: Path) -> dict[str, Any] | None:
    prd_path = Path(project_root) / "plans" / "beads-orchestrator-prd.md"
    if not prd_path.exists():
        return None

    try:
        text = prd_path.read_text(encoding="utf-8")
    except OSError as exc:
        log_json("WARN", "beads_prd_context_read_failed", details={"path": str(prd_path), "error": str(exc)})
        return None

    title = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break

    return {
        "path": str(prd_path),
        "title": title or prd_path.stem,
    }


def load_conductor_track_context(project_root: Path) -> dict[str, Any] | None:
    tracks_root = Path(project_root) / "conductor" / "tracks"
    if not tracks_root.exists():
        return None

    metadata_paths = sorted(tracks_root.glob("*/metadata.json"))
    if not metadata_paths:
        return None

    active_metadata: dict[str, Any] | None = None
    active_path: Path | None = None

    for metadata_path in metadata_paths:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        status = str(metadata.get("status", "")).lower()
        if status in {"active", "in_progress"}:
            active_metadata = metadata
            active_path = metadata_path
            break
        if active_metadata is None:
            active_metadata = metadata
            active_path = metadata_path

    if active_metadata is None or active_path is None:
        return None

    track_dir = active_path.parent
    context = dict(active_metadata)
    context.setdefault("track_id", track_dir.name)
    context["path"] = str(track_dir)
    return context


def build_beads_runtime_input(
    *,
    goal: str,
    project_root: Path,
    runtime_mode: str,
    goal_queue: Any = None,
    goal_archive: Any = None,
    active_goal: str | None = None,
    active_context: Mapping[str, Any] | None = None,
    goal_type: str | None = None,
) -> BeadsInput:
    return build_beads_input(
        goal=goal,
        goal_type=goal_type,
        project_root=Path(project_root),
        runtime_mode=runtime_mode,
        queue_summary=build_queue_summary(
            goal_queue,
            goal_archive,
            active_goal=active_goal or goal,
        ),
        active_context=active_context or {},
        prd_context=load_prd_context(Path(project_root)),
        conductor_track=load_conductor_track_context(Path(project_root)),
    )


def _result(
    *,
    ok: bool,
    status: str,
    decision: BeadsDecision | None = None,
    error: str | None = None,
    stderr: str | None = None,
    duration_ms: int = 0,
) -> BeadsResult:
    return {
        "schema_version": BEADS_SCHEMA_VERSION,
        "ok": ok,
        "status": status,
        "decision": decision,
        "error": error,
        "stderr": stderr,
        "duration_ms": duration_ms,
    }


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def validate_beads_result(payload: Any) -> BeadsResult:
    if not isinstance(payload, dict):
        raise ValueError("BEADS bridge payload must be a JSON object.")
    if payload.get("schema_version") != BEADS_SCHEMA_VERSION:
        raise ValueError("BEADS bridge schema version mismatch.")
    if not isinstance(payload.get("ok"), bool):
        raise ValueError("BEADS bridge payload missing boolean 'ok'.")
    if payload.get("status") not in {"ok", "error"}:
        raise ValueError("BEADS bridge payload has invalid status.")
    if not isinstance(payload.get("duration_ms"), int):
        raise ValueError("BEADS bridge payload missing integer duration_ms.")

    decision = payload.get("decision")
    if decision is not None:
        if not isinstance(decision, dict):
            raise ValueError("BEADS bridge decision must be an object.")
        if decision.get("schema_version") != BEADS_SCHEMA_VERSION:
            raise ValueError("BEADS decision schema version mismatch.")
        if decision.get("status") not in {"allow", "block", "revise"}:
            raise ValueError("BEADS decision status is invalid.")
        if not isinstance(decision.get("decision_id"), str) or not decision["decision_id"].strip():
            raise ValueError("BEADS decision requires a non-empty decision_id.")
        if not isinstance(decision.get("summary"), str):
            raise ValueError("BEADS decision requires a summary string.")
        for key in (
            "rationale",
            "required_constraints",
            "required_skills",
            "required_tests",
            "follow_up_goals",
        ):
            if not _is_string_list(decision.get(key)):
                raise ValueError(f"BEADS decision field '{key}' must be a list of strings.")
        stop_reason = decision.get("stop_reason")
        if stop_reason is not None and not isinstance(stop_reason, str):
            raise ValueError("BEADS decision stop_reason must be a string or null.")

    error = payload.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError("BEADS bridge error must be a string or null.")
    stderr = payload.get("stderr")
    if stderr is not None and not isinstance(stderr, str):
        raise ValueError("BEADS bridge stderr must be a string or null.")

    return payload  # type: ignore[return-value]


class BeadsBridge:
    """Adapter that invokes the BEADS Node bridge as a subprocess."""

    def __init__(self, config: BeadsBridgeConfig, *, project_root: Path) -> None:
        self.config = config
        self.project_root = Path(project_root)

    @classmethod
    def from_defaults(
        cls,
        project_root: Path,
        *,
        command: Sequence[str] | None = None,
        timeout_seconds: float = 20.0,
        enabled: bool = True,
        required: bool = False,
        persist_artifacts: bool = True,
        scope: str = "goal_run",
        env: Mapping[str, str] | None = None,
    ) -> "BeadsBridge":
        resolved_env = dict(env or {})
        if not any(resolved_env.get(key, "").strip() for key in ("BEADS_CLI", "BD_COMMAND")):
            resolved_cli = resolve_beads_cli(project_root)
            if resolved_cli != "bd":
                resolved_env["BEADS_CLI"] = resolved_cli
        cfg = BeadsBridgeConfig(
            command=tuple(command or default_beads_command(project_root)),
            timeout_seconds=timeout_seconds,
            enabled=enabled,
            required=required,
            persist_artifacts=persist_artifacts,
            scope=scope,
            env=resolved_env,
        )
        return cls(cfg, project_root=project_root)

    def run(self, payload: BeadsInput) -> BeadsResult:
        started_at = time.monotonic()
        command = list(self.config.command)
        log_json("INFO", "beads_bridge_start", details={"command": command, "scope": self.config.scope})

        env = os.environ.copy()
        env.update(self.config.env)

        try:
            proc = subprocess.run(
                command,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                cwd=self.project_root,
                env=env,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            log_json("WARN", "beads_bridge_missing", details={"error": str(exc), "command": command})
            return _result(ok=False, status="error", error="bridge_not_found", stderr=str(exc), duration_ms=duration_ms)
        except subprocess.TimeoutExpired as exc:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            log_json("WARN", "beads_bridge_timeout", details={"timeout_seconds": self.config.timeout_seconds})
            return _result(
                ok=False,
                status="error",
                error="timeout",
                stderr=(exc.stderr if isinstance(exc.stderr, str) else None),
                duration_ms=duration_ms,
            )

        duration_ms = int((time.monotonic() - started_at) * 1000)

        try:
            raw = json.loads(proc.stdout)
        except json.JSONDecodeError:
            if proc.returncode != 0:
                log_json(
                    "WARN",
                    "beads_bridge_process_error",
                    details={"returncode": proc.returncode, "stderr": proc.stderr.strip() or None},
                )
                return _result(
                    ok=False,
                    status="error",
                    error="process_error",
                    stderr=proc.stderr.strip() or None,
                    duration_ms=duration_ms,
                )
            log_json("WARN", "beads_bridge_invalid_json", details={"stdout_snippet": proc.stdout[:200]})
            return _result(
                ok=False,
                status="error",
                error="invalid_json",
                stderr=proc.stderr.strip() or None,
                duration_ms=duration_ms,
            )

        try:
            validated = validate_beads_result(raw)
        except ValueError as exc:
            log_json("WARN", "beads_bridge_invalid_contract", details={"error": str(exc)})
            return _result(
                ok=False,
                status="error",
                error="invalid_contract",
                stderr=proc.stderr.strip() or None,
                duration_ms=duration_ms,
            )

        validated["duration_ms"] = duration_ms
        if validated.get("stderr") is None:
            validated["stderr"] = proc.stderr.strip() or None
        if proc.returncode != 0:
            log_json(
                "WARN",
                "beads_bridge_structured_error",
                details={"returncode": proc.returncode, "error": validated.get("error")},
            )
        log_json(
            "INFO",
            "beads_bridge_complete",
            details={"ok": validated["ok"], "status": validated["status"], "duration_ms": duration_ms},
        )
        return validated

    def to_runtime_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "required": self.config.required,
            "scope": self.config.scope,
            "timeout_seconds": self.config.timeout_seconds,
            "command": list(self.config.command),
            "persist_artifacts": self.config.persist_artifacts,
        }
