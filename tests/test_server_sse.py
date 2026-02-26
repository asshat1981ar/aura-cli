import asyncio
import json
from fastapi.testclient import TestClient
from aura_cli import server

client = TestClient(server.app)


def test_tools_requires_auth_when_token_set(monkeypatch):
    monkeypatch.setenv("AGENT_API_TOKEN", "secret")
    with TestClient(server.app) as authed:
        r = authed.get("/tools")
        assert r.status_code == 401
        r = authed.get("/tools", headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200


def test_goal_sse_shape(monkeypatch):
    # disable auth for this test
    monkeypatch.delenv("AGENT_API_TOKEN", raising=False)
    # stub run_cycle to avoid real model calls
    monkeypatch.setattr(server.orchestrator, "run_cycle", lambda goal, dry_run=False: {
        "cycle_id": "stub-1",
        "phase_outputs": {"verification": {"status": "pass", "failures": []}},
        "stop_reason": "done",
    })

    with TestClient(server.app) as c:
        events = []
        with c.stream(
            "POST",
            "/execute",
            json={"tool_name": "goal", "args": ["sample goal"]},
            headers={"Accept": "text/event-stream"},
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if isinstance(line, bytes):
                    line = line.decode()
                if line.startswith("data: "):
                    payload = line[len("data: "):]
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError:
                        events.append(payload)
                if any(isinstance(e, dict) and e.get("type") == "complete" for e in events):
                    break
        assert any(e.get("type") == "start" for e in events if isinstance(e, dict))
        health = next(e for e in events if isinstance(e, dict) and e.get("type") == "health")
        assert health.get("status") == "ok"
        providers = health.get("providers", {})
        for key in ["openai", "openrouter", "gemini"]:
            assert key in providers
        assert any(e.get("type") == "cycle" for e in events if isinstance(e, dict))
        complete = next(e for e in events if isinstance(e, dict) and e.get("type") == "complete")
        assert "stop_reason" in complete


def test_run_sse_stream(monkeypatch):
    # disable auth and use a safe command
    monkeypatch.delenv("AGENT_API_TOKEN", raising=False)
    monkeypatch.setenv("AGENT_API_ENABLE_RUN", "1")
    cmd = "python -m site --user-site"
    with TestClient(server.app) as c:
        chunks = []
        with c.stream(
            "POST",
            "/execute",
            json={"tool_name": "run", "args": [cmd]},
            headers={"Accept": "text/event-stream"},
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if isinstance(line, bytes):
                    line = line.decode()
                if line.startswith("data: "):
                    payload = line[len("data: "):]
                    try:
                        chunks.append(json.loads(payload))
                    except json.JSONDecodeError:
                        continue
                if any(isinstance(e, dict) and e.get("type") == "exit" for e in chunks):
                    break
        assert any(e.get("type") == "stdout" for e in chunks if isinstance(e, dict))
        exit_evt = next(e for e in chunks if isinstance(e, dict) and e.get("type") == "exit")
        assert exit_evt.get("code") == 0


def test_run_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_API_TOKEN", raising=False)
    monkeypatch.delenv("AGENT_API_ENABLE_RUN", raising=False)
    with TestClient(server.app) as c:
        r = c.post("/execute", json={"tool_name": "run", "args": ["python -V"]})
    assert r.status_code == 403
