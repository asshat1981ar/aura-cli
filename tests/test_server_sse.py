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
    monkeypatch.setenv("AGENT_API_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setattr(server, "_beads_runtime_snapshot", lambda: {"enabled": True, "required": True, "scope": "goal_run"})
    # stub run_cycle to avoid real model calls
    monkeypatch.setattr(server.orchestrator, "run_cycle", lambda goal, dry_run=False: {
        "cycle_id": "stub-1",
        "goal": goal,
        "phase_outputs": {"verification": {"status": "pass", "failures": []}},
        "stop_reason": "done",
        "cycle_summary": {
            "cycle_id": "stub-1",
            "goal": goal,
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
        assert health["beads_runtime"] == {"enabled": True, "required": True, "scope": "goal_run"}
        cycle = next(e for e in events if isinstance(e, dict) and e.get("type") == "cycle")
        assert cycle["summary"]["cycle_id"] == "stub-1"
        assert cycle["summary"]["goal"] == "sample goal"
        assert cycle["summary"]["verification_status"] == "pass"
        assert cycle["summary"]["outcome"] == "SUCCESS"
        assert cycle["summary"]["beads_status"] == "allow"
        assert cycle["summary"]["beads_decision_id"] == "beads-1"
        complete = next(e for e in events if isinstance(e, dict) and e.get("type") == "complete")
        assert "stop_reason" in complete
        assert complete["history"][-1]["stop_reason"] == "done"


def test_run_sse_stream(monkeypatch):
    # disable auth and use a safe command
    monkeypatch.delenv("AGENT_API_TOKEN", raising=False)
    monkeypatch.setenv("AGENT_API_ALLOW_UNAUTHENTICATED", "1")
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
    monkeypatch.setenv("AGENT_API_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("AGENT_API_ENABLE_RUN", raising=False)
    with TestClient(server.app) as c:
        r = c.post("/execute", json={"tool_name": "run", "args": ["python -V"]})
    assert r.status_code == 403
