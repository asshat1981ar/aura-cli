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


# ---------------------------------------------------------------------------
# transport/health handler
# ---------------------------------------------------------------------------

def test_health_returns_ok(mock_orc):
    """transport/health must return status=ok and version=1.0."""
    t = StdioTransport(mock_orc)
    result = asyncio.run(t._handle_health({}))
    assert result["status"] == "ok"
    assert result["version"] == "1.0"


def test_health_has_uptime(mock_orc):
    """transport/health must include a non-negative numeric uptime_s field."""
    import time
    t = StdioTransport(mock_orc)
    time.sleep(0.01)
    result = asyncio.run(t._handle_health({}))
    assert "uptime_s" in result
    assert isinstance(result["uptime_s"], float)
    assert result["uptime_s"] >= 0.0


def test_health_via_jsonrpc(mock_orc, capsys):
    """End-to-end: transport/health over JSON-RPC returns result with status ok."""
    t = StdioTransport(mock_orc)
    asyncio.run(t._handle_line(json.dumps({
        "jsonrpc": "2.0", "id": 42, "method": "transport/health", "params": {}
    })))
    captured = capsys.readouterr()
    resp = json.loads(captured.out)
    assert resp["id"] == 42
    assert resp["result"]["status"] == "ok"


# ---------------------------------------------------------------------------
# goal/queue handler
# ---------------------------------------------------------------------------

def test_goal_queue_returns_list(mock_orc):
    """goal/queue must return pending list and count even with no goal_queue attr."""
    # Remove goal_queue so MagicMock doesn't auto-create it
    mock_orc_plain = MagicMock(spec=["agents", "run_loop"])
    t = StdioTransport(mock_orc_plain)
    result = asyncio.run(t._handle_goal_queue({}))
    assert "pending" in result
    assert isinstance(result["pending"], list)
    assert result["pending"] == []
    assert "count" in result
    assert result["count"] == 0


def test_goal_queue_reads_orchestrator_queue(mock_orc):
    """goal/queue returns goals from orchestrator.goal_queue when present."""
    from collections import deque
    mock_queue = MagicMock()
    mock_queue.queue = deque(["goal A", "goal B"])
    mock_orc.goal_queue = mock_queue
    t = StdioTransport(mock_orc)
    result = asyncio.run(t._handle_goal_queue({}))
    assert result["pending"] == ["goal A", "goal B"]
    assert result["count"] == 2
