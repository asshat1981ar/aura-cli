from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class RuntimeBootstrapState:
    runtime: dict[str, Any] = field(default_factory=dict)
    orchestrator: Any = None
    model_adapter: Any = None
    memory_store: Any = None
    runtime_init_error: str | None = None


def current_project_root(default_root: Path) -> Path:
    configured_root = os.getenv("AURA_PROJECT_ROOT")
    return Path(configured_root).resolve() if configured_root else default_root.resolve()


def run_db_migrations(log_json: Callable[..., None]) -> None:
    """Run auth and brain DB migrations idempotently before accepting requests."""
    try:
        from core.db_migrations import migrate_auth_db, migrate_brain_db  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        log_json("WARN", "aura_db_migrations_unavailable", details={"reason": "core.db_migrations not importable"})
        return

    try:
        from core.auth import _default_auth_db_path  # noqa: PLC0415
    except ImportError:  # pragma: no cover

        def _default_auth_db_path() -> Path:  # type: ignore[misc]
            custom = os.environ.get("AURA_AUTH_DB_PATH")
            if custom:
                return Path(custom)
            return Path.home() / ".local" / "share" / "aura" / "auth.db"

    auth_db_path = Path(os.environ["AURA_AUTH_DB_PATH"]) if os.environ.get("AURA_AUTH_DB_PATH") else _default_auth_db_path()
    brain_db_path = Path(os.environ.get("AURA_BRAIN_DB_PATH", "memory/brain_v2.db"))

    auth_versions = migrate_auth_db(auth_db_path)
    brain_versions = migrate_brain_db(brain_db_path)
    log_json(
        "INFO",
        "aura_db_migrations_applied",
        details={"auth_versions": auth_versions, "brain_versions": brain_versions},
    )


def apply_runtime_state(state: RuntimeBootstrapState, runtime_state: dict[str, Any], shared_state_module: Any) -> None:
    state.runtime = runtime_state
    state.orchestrator = runtime_state.get("orchestrator")
    state.model_adapter = runtime_state.get("model_adapter")
    state.memory_store = runtime_state.get("memory_store")

    shared_state_module.runtime = state.runtime
    shared_state_module.orchestrator = state.orchestrator
    shared_state_module.model_adapter = state.model_adapter
    shared_state_module.memory_store = state.memory_store
    shared_state_module._runtime_init_error = state.runtime_init_error


async def ensure_runtime_initialized(
    *,
    state: RuntimeBootstrapState,
    project_root: Path,
    create_runtime_func: Callable[..., dict[str, Any]],
    shared_state_module: Any,
    log_json: Callable[..., None],
) -> dict[str, Any]:
    if state.runtime:
        return state.runtime
    try:
        state.runtime_init_error = None
        runtime_state = await asyncio.to_thread(create_runtime_func, project_root, None)
        apply_runtime_state(state, runtime_state, shared_state_module)
        return runtime_state
    except Exception as exc:
        state.runtime_init_error = str(exc)
        shared_state_module._runtime_init_error = state.runtime_init_error
        log_json("WARN", "aura_server_runtime_init_failed", details={"error": state.runtime_init_error})
        return {}


async def resolve_runtime_component(
    *,
    state: RuntimeBootstrapState,
    name: str,
    ensure_runtime_initialized_func: Callable[[], Any],
) -> Any:
    component = getattr(state, name, None)
    if component is not None:
        return component
    if not state.runtime:
        await ensure_runtime_initialized_func()
        component = getattr(state, name, None)
        if component is not None:
            return component
    detail = f"{name} is not configured"
    if state.runtime_init_error and not state.runtime:
        detail = f"{detail}: {state.runtime_init_error}"
    from fastapi import HTTPException  # noqa: PLC0415

    raise HTTPException(status_code=503, detail=detail)


def runtime_metrics_snapshot(
    *,
    state: RuntimeBootstrapState,
    project_root: Path,
    list_registered_services_func: Callable[[], list[Any]],
    list_ai_environments_func: Callable[[Path], list[Any]],
    build_run_tool_audit_summary_func: Callable[[Any], Any],
) -> dict[str, Any]:
    entries: list[Any] = []
    if state.memory_store is not None and hasattr(state.memory_store, "read_log"):
        try:
            entries = list(state.memory_store.read_log(limit=1000) or [])
        except Exception:
            entries = []
    return {
        "total_calls": len(entries),
        "registered_services": len(list_registered_services_func()),
        "environment_count": len(list_ai_environments_func(project_root)),
        "run_tool_audit": build_run_tool_audit_summary_func(state.memory_store),
    }


def load_model_config_status(project_root: Path) -> bool:
    config_path = current_project_root(project_root) / "aura.config.json"
    try:
        if not config_path.exists():
            return False
        with open(config_path, encoding="utf-8") as handle:
            cfg = json.load(handle)
        return bool(cfg.get("model_name") or cfg.get("api_key") or cfg.get("openai_api_key") or cfg.get("local_model_profiles"))
    except Exception:
        return False
