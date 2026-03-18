"""Python bridge for invoking the Node-based BEADS adapter.

E5 enhancements:
- Retry with exponential backoff on transient failures
- Offline queue for persisting failed calls
- Last-result caching for conflict detection
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, List, Mapping, Sequence

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


_TRANSIENT_ERRORS = frozenset({"timeout", "process_error", "capability_unavailable"})


class BeadsBridge:
    """Adapter that invokes the BEADS Node bridge as a subprocess.

    E5 enhancements:
    - ``run_with_retry()`` — retries transient failures with backoff.
    - ``offline_queue`` — persists failed payloads for later replay.
    - ``last_result`` / ``has_conflict()`` — detect decision conflicts.
    """

    def __init__(self, config: BeadsBridgeConfig, *, project_root: Path) -> None:
        self.config = config
        self.project_root = Path(project_root)
        self.last_result: BeadsResult | None = None
        self._offline_queue_path = self.project_root / "memory" / "beads_offline_queue.jsonl"
        self._offline_dead_letter_path = self.project_root / "memory" / "beads_offline_queue.bad.jsonl"

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
            return _result(ok=False, status="error", error="capability_unavailable", stderr=str(exc), duration_ms=duration_ms)
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

    def run_with_retry(
        self,
        payload: BeadsInput,
        *,
        max_retries: int = 2,
        backoff_base: float = 1.0,
    ) -> BeadsResult:
        """Run with exponential backoff on transient failures.

        Retries on timeout, process_error, and capability_unavailable.
        Non-transient errors (invalid_json, invalid_contract) are returned immediately.
        On final failure, the payload is queued offline if persist_artifacts is enabled.
        """
        last_result: BeadsResult | None = None
        for attempt in range(1 + max_retries):
            result = self.run(payload)
            self.last_result = result

            if result["ok"]:
                return result

            error = result.get("error", "")
            if error not in _TRANSIENT_ERRORS:
                return result  # non-transient, don't retry

            last_result = result
            if attempt < max_retries:
                delay = backoff_base * (2 ** attempt)
                log_json("INFO", "beads_bridge_retry", details={
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "delay_s": delay,
                    "error": error,
                })
                time.sleep(delay)

        # All retries exhausted — queue for offline replay
        if self.config.persist_artifacts:
            self._enqueue_offline(payload)

        return last_result  # type: ignore[return-value]

    def _enqueue_offline(self, payload: BeadsInput) -> None:
        """Append a failed payload to the offline queue file."""
        try:
            self._offline_queue_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": time.time(),
                "payload": payload,
            }
            with open(self._offline_queue_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            log_json("INFO", "beads_offline_queued", details={
                "path": str(self._offline_queue_path),
                "goal": payload.get("goal", ""),
            })
        except OSError as exc:
            log_json("WARN", "beads_offline_queue_failed", details={"error": str(exc)})

    def _load_offline_entries(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load valid offline entries and quarantine malformed rows separately."""
        valid_entries: list[dict[str, Any]] = []
        invalid_entries: list[dict[str, Any]] = []

        try:
            with open(self._offline_queue_path, "r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as exc:
                        invalid_entries.append(
                            {
                                "ts": time.time(),
                                "line_number": line_number,
                                "error": "invalid_json",
                                "detail": str(exc),
                                "raw_line": line,
                            }
                        )
                        continue

                    payload = entry.get("payload") if isinstance(entry, dict) else None
                    if not isinstance(entry, dict) or not isinstance(payload, dict):
                        invalid_entries.append(
                            {
                                "ts": time.time(),
                                "line_number": line_number,
                                "error": "invalid_payload",
                                "detail": "offline queue entry must be an object with a dict payload",
                                "raw_line": line,
                            }
                        )
                        continue

                    valid_entries.append(entry)
        except OSError as exc:
            log_json("WARN", "beads_offline_queue_read_failed", details={"error": str(exc)})
            return [], []

        return valid_entries, invalid_entries

    def _write_dead_letters(self, entries: list[dict[str, Any]]) -> None:
        """Append malformed offline rows to a dead-letter log for later inspection."""
        if not entries:
            return
        try:
            self._offline_dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._offline_dead_letter_path, "a", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry) + "\n")
            log_json(
                "WARN",
                "beads_offline_dead_lettered",
                details={"count": len(entries), "path": str(self._offline_dead_letter_path)},
            )
        except OSError as exc:
            log_json("WARN", "beads_offline_dead_letter_failed", details={"error": str(exc)})

    def replay_offline_queue(self) -> List[BeadsResult]:
        """Replay all queued payloads, returning results. Clears queue on success."""
        if not self._offline_queue_path.exists():
            return []

        entries, invalid_entries = self._load_offline_entries()
        self._write_dead_letters(invalid_entries)

        if not entries and not invalid_entries:
            return []

        results: List[BeadsResult] = []
        remaining: list = []
        for entry in entries:
            payload = entry.get("payload", {})
            result = self.run(payload)
            results.append(result)
            if not result["ok"]:
                remaining.append(entry)

        # Rewrite queue with only failed entries
        try:
            if remaining:
                with open(self._offline_queue_path, "w", encoding="utf-8") as f:
                    for entry in remaining:
                        f.write(json.dumps(entry) + "\n")
            else:
                self._offline_queue_path.unlink(missing_ok=True)
            log_json("INFO", "beads_offline_replay_complete", details={
                "total": len(entries),
                "succeeded": len(entries) - len(remaining),
                "remaining": len(remaining),
                "dead_lettered": len(invalid_entries),
            })
        except OSError as exc:
            log_json("WARN", "beads_offline_queue_write_failed", details={"error": str(exc)})

        return results

    def has_conflict(self, new_result: BeadsResult) -> bool:
        """Check if a new result's decision conflicts with the last cached result.

        A conflict occurs when both results have decisions but they disagree
        on the status (allow vs block vs revise).
        """
        if self.last_result is None:
            return False
        old_decision = self.last_result.get("decision")
        new_decision = new_result.get("decision")
        if old_decision is None or new_decision is None:
            return False
        return old_decision.get("status") != new_decision.get("status")

    def offline_queue_size(self) -> int:
        """Return the number of entries in the offline queue."""
        if not self._offline_queue_path.exists():
            return 0
        try:
            with open(self._offline_queue_path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0

    def offline_dead_letter_size(self) -> int:
        """Return the number of malformed offline entries that were quarantined."""
        if not self._offline_dead_letter_path.exists():
            return 0
        try:
            with open(self._offline_dead_letter_path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0

    def to_runtime_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "required": self.config.required,
            "scope": self.config.scope,
            "timeout_seconds": self.config.timeout_seconds,
            "command": list(self.config.command),
            "persist_artifacts": self.config.persist_artifacts,
            "offline_queue_size": self.offline_queue_size(),
            "offline_dead_letter_size": self.offline_dead_letter_size(),
        }
