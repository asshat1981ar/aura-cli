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
