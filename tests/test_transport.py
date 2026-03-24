"""Tests for core/transport.py — JSON-RPC 2.0 over stdio transport."""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from core.transport import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    METHOD_MAP,
    JSONRPCError,
    StdioTransport,
)


def _transport() -> StdioTransport:
    """Return a transport pointed at the current directory."""
    return StdioTransport(project_root=Path("."))


def _make_request(method: str, params: dict | None = None, req_id: object = "req-1") -> str:
    return json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}, "id": req_id})


def _parse_response(line: str) -> dict:
    return json.loads(line)


# ---------------------------------------------------------------------------
# Unit tests for handle_raw_line
# ---------------------------------------------------------------------------


class TestHandleRawLine(unittest.TestCase):
    """Tests for StdioTransport.handle_raw_line (synchronous, no I/O)."""

    def setUp(self) -> None:
        self.transport = _transport()

    # ------------------------------------------------------------------
    # Protocol-level errors
    # ------------------------------------------------------------------

    def test_parse_error_on_invalid_json(self) -> None:
        resp = self.transport.handle_raw_line("not valid json{{{")
        self.assertEqual(resp["jsonrpc"], "2.0")
        self.assertIsNone(resp["id"])
        self.assertEqual(resp["error"]["code"], PARSE_ERROR)

    def test_invalid_request_non_object(self) -> None:
        resp = self.transport.handle_raw_line(json.dumps([1, 2, 3]))
        self.assertEqual(resp["error"]["code"], INVALID_REQUEST)

    def test_invalid_request_missing_jsonrpc(self) -> None:
        resp = self.transport.handle_raw_line(json.dumps({"method": "goal.add", "id": 1}))
        self.assertEqual(resp["error"]["code"], INVALID_REQUEST)

    def test_invalid_request_wrong_jsonrpc_version(self) -> None:
        resp = self.transport.handle_raw_line(
            json.dumps({"jsonrpc": "1.0", "method": "goal.add", "id": 1})
        )
        self.assertEqual(resp["error"]["code"], INVALID_REQUEST)

    def test_invalid_request_missing_method(self) -> None:
        resp = self.transport.handle_raw_line(json.dumps({"jsonrpc": "2.0", "id": 1}))
        self.assertEqual(resp["error"]["code"], INVALID_REQUEST)

    def test_invalid_request_rejects_object_id(self) -> None:
        req = json.dumps({"jsonrpc": "2.0", "method": "goal.add", "id": {"bad": "id"}})
        resp = self.transport.handle_raw_line(req)
        self.assertEqual(resp["error"]["code"], INVALID_REQUEST)
        self.assertIsNone(resp["id"])

    def test_method_not_found(self) -> None:
        resp = self.transport.handle_raw_line(_make_request("unknown.method"))
        self.assertEqual(resp["error"]["code"], METHOD_NOT_FOUND)
        self.assertIn("unknown.method", resp["error"]["message"])
        self.assertEqual(resp["id"], "req-1")

    def test_invalid_params_non_object(self) -> None:
        req = json.dumps({"jsonrpc": "2.0", "method": "goal.add", "params": [1, 2], "id": "x"})
        resp = self.transport.handle_raw_line(req)
        self.assertEqual(resp["error"]["code"], INVALID_PARAMS)

    # ------------------------------------------------------------------
    # Successful dispatch (mocked)
    # ------------------------------------------------------------------

    def _make_mock_registry(self, exit_code: int = 0, output: str = "ok\n") -> dict:
        """Build a fake COMMAND_DISPATCH_REGISTRY that returns a fixed exit code."""
        handler = MagicMock(return_value=exit_code)
        # Make the handler write *output* to stdout so capture works.
        def _side_effect(ctx):
            print(output, end="")
            return exit_code
        handler.side_effect = _side_effect

        rule = SimpleNamespace(
            action="goal_add",
            requires_runtime=False,
            handler=handler,
        )
        return {"goal_add": rule}

    def test_successful_dispatch_returns_exit_code_and_output(self) -> None:
        registry = self._make_mock_registry(exit_code=0, output="Added goal.\n")
        with patch("core.transport.StdioTransport._dispatch", return_value=(0, "Added goal.\n")):
            resp = self.transport.handle_raw_line(_make_request("goal.add", {"add_goal": "Test"}))
        self.assertNotIn("error", resp)
        self.assertEqual(resp["result"]["code"], 0)
        self.assertIn("Added goal.", resp["result"]["cli_output"])

    def test_id_is_preserved_in_response(self) -> None:
        with patch("core.transport.StdioTransport._dispatch", return_value=(0, "")):
            resp = self.transport.handle_raw_line(_make_request("goal.add", req_id="my-unique-id"))
        self.assertEqual(resp["id"], "my-unique-id")

    def test_null_id_preserved(self) -> None:
        req = json.dumps({"jsonrpc": "2.0", "method": "goal.status", "params": {}, "id": None})
        with patch("core.transport.StdioTransport._dispatch", return_value=(0, "")):
            resp = self.transport.handle_raw_line(req)
        self.assertIsNone(resp["id"])

    def test_integer_id_preserved(self) -> None:
        req = json.dumps({"jsonrpc": "2.0", "method": "goal.status", "params": {}, "id": 42})
        with patch("core.transport.StdioTransport._dispatch", return_value=(0, "")):
            resp = self.transport.handle_raw_line(req)
        self.assertEqual(resp["id"], 42)

    def test_internal_error_wraps_unexpected_exceptions(self) -> None:
        with patch("core.transport.StdioTransport._dispatch", side_effect=RuntimeError("oops")):
            resp = self.transport.handle_raw_line(_make_request("goal.add"))
        self.assertEqual(resp["error"]["code"], INTERNAL_ERROR)
        self.assertIn("oops", resp["error"]["message"])

    def test_non_zero_exit_code_is_success_response(self) -> None:
        """Handler returning non-zero exit code is not a protocol error."""
        with patch("core.transport.StdioTransport._dispatch", return_value=(1, "fail\n")):
            resp = self.transport.handle_raw_line(_make_request("goal.run"))
        self.assertNotIn("error", resp)
        self.assertEqual(resp["result"]["code"], 1)

    # ------------------------------------------------------------------
    # params defaults / absent params
    # ------------------------------------------------------------------

    def test_absent_params_defaults_to_empty(self) -> None:
        req = json.dumps({"jsonrpc": "2.0", "method": "goal.status", "id": "r"})
        with patch("core.transport.StdioTransport._dispatch", return_value=(0, "")) as mock_dispatch:
            self.transport.handle_raw_line(req)
        mock_dispatch.assert_called_once_with("goal.status", {})

    def test_null_params_treated_as_empty(self) -> None:
        req = json.dumps({"jsonrpc": "2.0", "method": "goal.status", "params": None, "id": "r"})
        with patch("core.transport.StdioTransport._dispatch", return_value=(0, "")) as mock_dispatch:
            self.transport.handle_raw_line(req)
        mock_dispatch.assert_called_once_with("goal.status", {})

    def test_notification_returns_no_response(self) -> None:
        req = json.dumps({"jsonrpc": "2.0", "method": "goal.status", "params": {}})
        with patch("core.transport.StdioTransport._dispatch", return_value=(0, "")):
            resp = self.transport.handle_raw_line(req)
        self.assertIsNone(resp)


# ---------------------------------------------------------------------------
# Tests for _dispatch (verifies stdout capture and namespace building)
# ---------------------------------------------------------------------------


class TestDispatchCapture(unittest.TestCase):
    """Tests that _dispatch captures stdout/stderr and doesn't corrupt JSON stream."""

    def setUp(self) -> None:
        self.transport = _transport()

    def test_stdout_from_handler_is_captured(self) -> None:
        """print() calls inside the handler must end up in cli_output, not on real stdout."""
        def _fake_handler(ctx):
            print("captured output")
            return 0

        rule = SimpleNamespace(action="doctor", requires_runtime=False, handler=_fake_handler)
        registry = {"doctor": rule}

        with patch("aura_cli.cli_main.COMMAND_DISPATCH_REGISTRY", registry), \
             patch("aura_cli.cli_main._prepare_runtime_context"):
            code, output = self.transport._dispatch("system.doctor", {})

        self.assertEqual(code, 0)
        self.assertIn("captured output", output)

    def test_stderr_from_handler_is_captured(self) -> None:
        def _fake_handler(ctx):
            import sys as _sys
            print("err msg", file=_sys.stderr)
            return 0

        rule = SimpleNamespace(action="doctor", requires_runtime=False, handler=_fake_handler)

        with patch("aura_cli.cli_main.COMMAND_DISPATCH_REGISTRY", {"doctor": rule}), \
             patch("aura_cli.cli_main._prepare_runtime_context"):
            code, output = self.transport._dispatch("system.doctor", {})

        self.assertEqual(code, 0)
        self.assertIn("err msg", output)

    def test_dispatch_raises_for_unknown_method(self) -> None:
        with self.assertRaises(JSONRPCError) as cm:
            self.transport._dispatch("no.such.method", {})
        self.assertEqual(cm.exception.rpc_code, METHOD_NOT_FOUND)

    def test_dispatch_populates_handler_defaults(self) -> None:
        def _fake_handler(ctx):
            self.assertEqual(ctx.args.mcp_args, None)
            self.assertEqual(ctx.args.json, True)
            self.assertEqual(ctx.args.dry_run, False)
            return 0

        rule = SimpleNamespace(action="mcp_call", requires_runtime=False, handler=_fake_handler)

        with patch("aura_cli.cli_main.COMMAND_DISPATCH_REGISTRY", {"mcp_call": rule}), \
             patch("aura_cli.cli_main._prepare_runtime_context"):
            code, _output = self.transport._dispatch("mcp.call", {"mcp_call": "demo.tool"})

        self.assertEqual(code, 0)


# ---------------------------------------------------------------------------
# Tests for _send
# ---------------------------------------------------------------------------


class TestSend(unittest.TestCase):
    def test_send_writes_json_line_to_real_stdout(self) -> None:
        buf = io.StringIO()
        transport = StdioTransport(project_root=Path("."))
        transport._real_stdout = buf
        transport._send({"jsonrpc": "2.0", "result": {"code": 0}, "id": "x"})
        line = buf.getvalue()
        self.assertTrue(line.endswith("\n"))
        parsed = json.loads(line)
        self.assertEqual(parsed["id"], "x")

    def test_notification_is_not_sent(self) -> None:
        buf = io.StringIO()
        transport = StdioTransport(project_root=Path("."))
        transport._real_stdout = buf
        response = transport.handle_raw_line(json.dumps({"jsonrpc": "2.0", "method": "goal.status"}))
        if response is not None:
            transport._send(response)
        self.assertEqual(buf.getvalue(), "")


# ---------------------------------------------------------------------------
# Tests for METHOD_MAP completeness
# ---------------------------------------------------------------------------


class TestMethodMap(unittest.TestCase):
    def test_all_mapped_actions_exist_in_registry(self) -> None:
        """Every value in METHOD_MAP must be a key in COMMAND_DISPATCH_REGISTRY."""
        from aura_cli.cli_main import COMMAND_DISPATCH_REGISTRY  # pylint: disable=import-outside-toplevel

        missing = [
            (method, action)
            for method, action in METHOD_MAP.items()
            if action not in COMMAND_DISPATCH_REGISTRY
        ]
        self.assertEqual(
            missing,
            [],
            msg=f"These METHOD_MAP entries have no matching COMMAND_DISPATCH_REGISTRY key: {missing}",
        )

    def test_method_map_keys_use_dot_notation(self) -> None:
        for method in METHOD_MAP:
            self.assertIn(".", method, msg=f"Method '{method}' does not use dot-notation")


# ---------------------------------------------------------------------------
# Integration: main() intercepts --stdio-rpc before argparse
# ---------------------------------------------------------------------------


class TestMainStdioRpcFlag(unittest.TestCase):
    """Verify that --stdio-rpc is intercepted early in main()."""

    def _run_main_with_flag(self, argv: list[str]) -> int:
        """Run cli_main.main() with a fake StdioTransport injected."""
        import aura_cli.cli_main as cli_main  # pylint: disable=import-outside-toplevel

        mock_transport = MagicMock()
        mock_transport_cls = MagicMock(return_value=mock_transport)

        fake_transport_mod = ModuleType("core.transport")
        fake_transport_mod.StdioTransport = mock_transport_cls  # type: ignore[attr-defined]

        original_modules = dict(sys.modules)
        sys.modules["core.transport"] = fake_transport_mod
        try:
            rc = cli_main.main(argv=argv, project_root_override=".")
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)

        return rc, mock_transport_cls, mock_transport

    def test_stdio_rpc_flag_launches_transport(self) -> None:
        import aura_cli.cli_main as cli_main  # pylint: disable=import-outside-toplevel

        with patch("aura_cli.cli_main.parse_cli_args") as mock_parse:
            rc, mock_cls, mock_transport = self._run_main_with_flag(["--stdio-rpc"])

        self.assertEqual(rc, 0)
        mock_cls.assert_called_once()
        mock_transport.run.assert_called_once()
        mock_parse.assert_not_called()

    def test_stdio_rpc_skips_argparse_entirely(self) -> None:
        """Additional flags alongside --stdio-rpc must not be parsed by argparse."""
        import aura_cli.cli_main as cli_main  # pylint: disable=import-outside-toplevel

        with patch("aura_cli.cli_main.parse_cli_args") as mock_parse:
            rc, _cls, _transport = self._run_main_with_flag(["--stdio-rpc", "--json"])

        self.assertEqual(rc, 0)
        mock_parse.assert_not_called()

    def test_stdio_rpc_transport_receives_project_root(self) -> None:
        """StdioTransport should be constructed with the resolved project_root."""
        _rc, mock_cls, _transport = self._run_main_with_flag(["--stdio-rpc"])
        call_kwargs = mock_cls.call_args.kwargs
        self.assertIn("project_root", call_kwargs)


if __name__ == "__main__":
    unittest.main()
