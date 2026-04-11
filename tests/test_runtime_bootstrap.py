from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from aura_cli.api.runtime_bootstrap import (
    RuntimeBootstrapState,
    apply_runtime_state,
    ensure_runtime_initialized,
    resolve_runtime_component,
)


def _shared_state() -> SimpleNamespace:
    return SimpleNamespace(
        runtime={},
        orchestrator=None,
        model_adapter=None,
        memory_store=None,
        _runtime_init_error="stale",
    )


def _log_json(*_args, **_kwargs) -> None:
    return None


def test_apply_runtime_state_syncs_shared_error_state():
    state = RuntimeBootstrapState(runtime_init_error="boot failed")
    orchestrator = object()
    shared_state = _shared_state()

    apply_runtime_state(
        state,
        {
            "orchestrator": orchestrator,
            "model_adapter": "model",
            "memory_store": "memory",
        },
        shared_state,
    )

    assert shared_state.runtime == state.runtime
    assert shared_state.orchestrator is orchestrator
    assert shared_state.model_adapter == "model"
    assert shared_state.memory_store == "memory"
    assert shared_state._runtime_init_error == "boot failed"


@pytest.mark.asyncio
async def test_ensure_runtime_initialized_clears_shared_error_on_success():
    state = RuntimeBootstrapState(runtime_init_error="old failure")
    orchestrator = object()
    shared_state = _shared_state()

    def create_runtime(_project_root: Path, _config):
        return {
            "orchestrator": orchestrator,
            "model_adapter": "model",
            "memory_store": "memory",
        }

    result = await ensure_runtime_initialized(
        state=state,
        project_root=Path("/tmp/project"),
        create_runtime_func=create_runtime,
        shared_state_module=shared_state,
        log_json=_log_json,
    )

    assert result["orchestrator"] is orchestrator
    assert state.runtime_init_error is None
    assert shared_state._runtime_init_error is None


@pytest.mark.asyncio
async def test_ensure_runtime_initialized_publishes_failure_to_shared_state():
    state = RuntimeBootstrapState()
    shared_state = _shared_state()

    def create_runtime(_project_root: Path, _config):
        raise RuntimeError("runtime boot failed")

    result = await ensure_runtime_initialized(
        state=state,
        project_root=Path("/tmp/project"),
        create_runtime_func=create_runtime,
        shared_state_module=shared_state,
        log_json=_log_json,
    )

    assert result == {}
    assert state.runtime_init_error == "runtime boot failed"
    assert shared_state._runtime_init_error == "runtime boot failed"


@pytest.mark.asyncio
async def test_resolve_runtime_component_invalid_name_returns_503():
    state = RuntimeBootstrapState(runtime_init_error="runtime boot failed")

    async def ensure_runtime_initialized_func():
        return {}

    with pytest.raises(HTTPException, match="not_real is not configured: runtime boot failed"):
        await resolve_runtime_component(
            state=state,
            name="not_real",
            ensure_runtime_initialized_func=ensure_runtime_initialized_func,
        )
