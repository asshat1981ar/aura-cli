import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock
from core.transport import StdioTransport

@pytest.fixture
def mock_orc():
    orc = MagicMock()
    orc.agents = {"act": MagicMock(), "plan": MagicMock()}
    orc.run_loop = MagicMock(return_value="done")
    return orc

def test_ping(mock_orc):
    t = StdioTransport(mock_orc)
    result = asyncio.run(t._handle_ping({}))
    assert result == {"pong": True}

def test_status(mock_orc):
    t = StdioTransport(mock_orc)
    result = asyncio.run(t._handle_status({}))
    assert result["agent_count"] == 2

def test_unknown_method_returns_error(mock_orc, capsys):
    t = StdioTransport(mock_orc)
    asyncio.run(t._handle_line(json.dumps({"jsonrpc":"2.0","id":1,"method":"unknown","params":{}})))
    captured = capsys.readouterr()
    resp = json.loads(captured.out)
    assert resp["error"]["code"] == -32601

def test_parse_error(mock_orc, capsys):
    t = StdioTransport(mock_orc)
    asyncio.run(t._handle_line("not json"))
    captured = capsys.readouterr()
    resp = json.loads(captured.out)
    assert resp["error"]["code"] == -32700

def test_goal_calls_orchestrator(mock_orc):
    t = StdioTransport(mock_orc)
    result = asyncio.run(t._handle_goal({"goal": "add tests", "dry_run": True}))
    mock_orc.run_loop.assert_called_once_with("add tests", dry_run=True, context_injection=None)
    assert result["status"] == "ok"

def test_notification_no_response(mock_orc, capsys):
    """JSON-RPC 2.0: notifications (no id field) must not receive a response."""
    t = StdioTransport(mock_orc)
    asyncio.run(t._handle_line(json.dumps({"jsonrpc": "2.0", "method": "ping", "params": {}})))
    captured = capsys.readouterr()
    assert captured.out == "", "Server MUST NOT reply to a notification"

def test_notification_unknown_method_silent(mock_orc, capsys):
    """Unknown method on a notification should be silently ignored."""
    t = StdioTransport(mock_orc)
    asyncio.run(t._handle_line(json.dumps({"jsonrpc": "2.0", "method": "nonexistent", "params": {}})))
    captured = capsys.readouterr()
    assert captured.out == ""


def test_write_notification_no_id(mock_orc, capsys):
    t = StdioTransport(mock_orc)
    t._write_notification("test/event", {"key": "val"})
    captured = capsys.readouterr()
    msg = json.loads(captured.out)
    assert "id" not in msg
    assert msg["method"] == "test/event"
    assert msg["params"] == {"key": "val"}


def test_goal_stream_sends_three_notifications(mock_orc, capsys):
    t = StdioTransport(mock_orc)
    result = asyncio.run(t._handle_goal_stream({"goal": "add tests", "dry_run": True}))
    captured = capsys.readouterr()
    lines = [l for l in captured.out.strip().split("\n") if l]
    assert len(lines) == 3  # starting, running, complete
    phases = [json.loads(l)["params"]["phase"] for l in lines]
    assert phases == ["starting", "running", "complete"]


def test_goal_stream_calls_orchestrator(mock_orc, capsys):
    t = StdioTransport(mock_orc)
    asyncio.run(t._handle_goal_stream({"goal": "refactor auth", "dry_run": False}))
    mock_orc.run_loop.assert_called_once_with("refactor auth", dry_run=False, context_injection=None)


def test_goal_stream_error_notification(capsys):
    orc = MagicMock()
    orc.run_loop = MagicMock(side_effect=RuntimeError("boom"))
    t = StdioTransport(orc)
    with pytest.raises(RuntimeError):
        asyncio.run(t._handle_goal_stream({"goal": "fail", "dry_run": True}))
    captured = capsys.readouterr()
    lines = [json.loads(l) for l in captured.out.strip().split("\n") if l]
    phases = [l["params"]["phase"] for l in lines]
    assert "error" in phases
