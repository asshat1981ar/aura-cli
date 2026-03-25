from __future__ import annotations

import asyncio
import importlib
import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException


def _load_server_module():
    fake_rt = {
        "orchestrator": MagicMock(),
        "model_adapter": MagicMock(),
        "memory_store": MagicMock(),
        "config_api_key": None,
    }
    fake_rt["orchestrator"].policy = MagicMock()
    sys.modules.pop("aura_cli.server", None)
    with patch("aura_cli.cli_main.create_runtime", return_value=fake_rt) as mock_create_runtime:
        module = importlib.import_module("aura_cli.server")
        mock_create_runtime.assert_not_called()
        module.runtime = fake_rt
        module.orchestrator = fake_rt["orchestrator"]
        module.model_adapter = fake_rt["model_adapter"]
        module.memory_store = fake_rt["memory_store"]
        return module


@pytest.fixture(scope="module")
def server_module():
    return _load_server_module()


@pytest.fixture(autouse=True)
def immediate_to_thread(server_module, monkeypatch):
    async def _immediate(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(server_module.asyncio, "to_thread", _immediate)


def _run(coro):
    return asyncio.run(coro)


async def _collect_streaming_response(response) -> list[str]:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        chunks.append(chunk)
    return chunks


def _sse_payloads(chunks: list[str]) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for chunk in chunks:
        if chunk.startswith("data: "):
            payloads.append(json.loads(chunk[len("data: ") :]))
    return payloads


def test_tools_requires_auth_when_token_set(monkeypatch, server_module):
    monkeypatch.setenv("AGENT_API_TOKEN", "secret")
    assert server_module.require_auth("Bearer secret") is None
    with pytest.raises(HTTPException):
        server_module.require_auth("Bearer wrong")


def test_goal_sse_shape(monkeypatch, server_module):
    monkeypatch.delenv("AGENT_API_TOKEN", raising=False)
    monkeypatch.setattr(server_module, "_beads_runtime_snapshot", lambda: {"enabled": True, "required": True, "scope": "goal_run"})
    to_thread_spy = MagicMock(side_effect=lambda func, /, *args, **kwargs: func(*args, **kwargs))

    async def _immediate(func, /, *args, **kwargs):
        return to_thread_spy(func, *args, **kwargs)

    monkeypatch.setattr(server_module.asyncio, "to_thread", _immediate)
    server_module.orchestrator.run_cycle = MagicMock(
        return_value={
            "cycle_id": "stub-1",
            "goal": "sample goal",
            "phase_outputs": {"verification": {"status": "pass", "failures": []}},
            "stop_reason": "done",
            "cycle_summary": {
                "cycle_id": "stub-1",
                "goal": "sample goal",
                "goal_type": "default",
                "state": "complete",
                "current_phase": "verify",
                "phase_status": {"verify": "pass"},
                "verification_status": "pass",
                "outcome": "SUCCESS",
                "stop_reason": "done",
                "failures": [],
                "retry_count": 0,
                "applied_files": [],
                "failed_files": [],
                "queued_follow_up_goals": [],
                "beads_status": "allow",
                "beads_decision_id": "beads-1",
                "beads_summary": "Proceed with the scoped change.",
                "beads_required_constraints": ["Keep CLI JSON stable."],
                "beads_follow_up_goals": ["Review telemetry"],
                "started_at": 1.0,
                "completed_at": 2.0,
                "duration_s": 1.0,
            },
        }
    )

    response = _run(server_module.execute(server_module.ExecuteRequest(tool_name="goal", args=["sample goal"])))
    chunks = _run(_collect_streaming_response(response))
    payloads = [json.loads(chunk[len("data: ") :]) for chunk in chunks if chunk.startswith("data: ")]
    assert any(evt.get("type") == "start" for evt in payloads if isinstance(evt, dict))
    health = next(evt for evt in payloads if isinstance(evt, dict) and evt.get("type") == "health")
    assert health.get("status") == "ok"
    for key in ["openai", "openrouter", "gemini"]:
        assert key in health.get("providers", {})
    assert health["beads_runtime"] == {"enabled": True, "required": True, "scope": "goal_run"}
    cycle = next(evt for evt in payloads if isinstance(evt, dict) and evt.get("type") == "cycle")
    assert cycle["summary"]["cycle_id"] == "stub-1"
    assert cycle["summary"]["goal"] == "sample goal"
    assert cycle["summary"]["verification_status"] == "pass"
    assert cycle["summary"]["outcome"] == "SUCCESS"
    assert cycle["summary"]["beads_status"] == "allow"
    assert cycle["summary"]["beads_decision_id"] == "beads-1"
    complete = next(evt for evt in payloads if isinstance(evt, dict) and evt.get("type") == "complete")
    assert "stop_reason" in complete
    assert complete["history"][-1]["stop_reason"] == "done"
    to_thread_spy.assert_called_once()
    assert to_thread_spy.call_args.args[0] is server_module.orchestrator.run_cycle


def test_run_sse_stream(monkeypatch, server_module):
    monkeypatch.delenv("AGENT_API_TOKEN", raising=False)
    monkeypatch.setenv("AGENT_API_ENABLE_RUN", "1")
    cmd = f"{sys.executable} -c \"import sys; print('stdout-line'); sys.stderr.write('stderr-line\\\\n')\""
    response = _run(server_module.execute(server_module.ExecuteRequest(tool_name="run", args=[cmd])))
    payloads = _sse_payloads(_run(_collect_streaming_response(response)))
    assert any(evt.get("type") == "stdout" and "stdout-line" in evt.get("data", "") for evt in payloads)
    assert any(evt.get("type") == "stderr" and "stderr-line" in evt.get("data", "") for evt in payloads)
    exit_evt = next(evt for evt in payloads if isinstance(evt, dict) and evt.get("type") == "exit")
    assert exit_evt.get("code") == 0


def test_run_disabled_by_default(monkeypatch, server_module):
    monkeypatch.delenv("AGENT_API_TOKEN", raising=False)
    monkeypatch.delenv("AGENT_API_ENABLE_RUN", raising=False)
    with pytest.raises(HTTPException) as exc:
        _run(server_module.execute(server_module.ExecuteRequest(tool_name="run", args=["python -V"])))
    assert exc.value.status_code == 403
