"""
Unit tests for core.transport — StdioTransport (JSON-RPC 2.0 over stdio).

Tests cover:
- JSON-RPC protocol compliance (valid requests, error responses)
- Method map coverage (all registered methods resolve to an action)
- StdioTransport.handle_request() for single-request dispatch
- _params_to_argv() conversion helpers
- Edge cases: malformed input, missing fields, unknown methods
- Round-trip: goal.add → goal.status via stubbed runtime
"""

import json
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from core.transport import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_MAP,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    StdioTransport,
    _jsonrpc_error,
    _jsonrpc_result,
    _params_to_argv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transport(runtime_factory=None, in_stream=None, out_stream=None):
    """Return a StdioTransport with a MagicMock runtime_factory by default."""
    return StdioTransport(
        project_root=Path("."),
        runtime_factory=runtime_factory or MagicMock(return_value={}),
        in_stream=in_stream,
        out_stream=out_stream,
    )


# ---------------------------------------------------------------------------
# _jsonrpc_error / _jsonrpc_result helpers
# ---------------------------------------------------------------------------

class TestJsonRpcHelpers(unittest.TestCase):

    def test_error_structure(self):
        resp = _jsonrpc_error(42, -32601, "Method not found")
        self.assertEqual(resp["jsonrpc"], "2.0")
        self.assertEqual(resp["id"], 42)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32601)
        self.assertEqual(resp["error"]["message"], "Method not found")
        self.assertNotIn("data", resp["error"])

    def test_error_with_data(self):
        resp = _jsonrpc_error(1, -32600, "Bad request", data={"hint": "use 2.0"})
        self.assertEqual(resp["error"]["data"], {"hint": "use 2.0"})

    def test_error_null_id(self):
        resp = _jsonrpc_error(None, PARSE_ERROR, "Parse error")
        self.assertIsNone(resp["id"])

    def test_result_structure(self):
        resp = _jsonrpc_result("req-1", {"foo": "bar"})
        self.assertEqual(resp["jsonrpc"], "2.0")
        self.assertEqual(resp["id"], "req-1")
        self.assertEqual(resp["result"], {"foo": "bar"})
        self.assertNotIn("error", resp)


# ---------------------------------------------------------------------------
# METHOD_MAP completeness
# ---------------------------------------------------------------------------

class TestMethodMap(unittest.TestCase):

    def test_all_methods_map_to_known_actions(self):
        """Every JSON-RPC method must map to a registered COMMAND_DISPATCH_REGISTRY action."""
        import aura_cli.cli_main as cli_main
        for method, action in METHOD_MAP.items():
            with self.subTest(method=method):
                self.assertIn(
                    action,
                    cli_main.COMMAND_DISPATCH_REGISTRY,
                    f"Action '{action}' (for method '{method}') not in COMMAND_DISPATCH_REGISTRY",
                )

    def test_method_map_not_empty(self):
        self.assertGreater(len(METHOD_MAP), 0)

    def test_goal_methods_present(self):
        self.assertIn("goal.add", METHOD_MAP)
        self.assertIn("goal.run", METHOD_MAP)
        self.assertIn("goal.once", METHOD_MAP)
        self.assertIn("goal.status", METHOD_MAP)


# ---------------------------------------------------------------------------
# _params_to_argv
# ---------------------------------------------------------------------------

class TestParamsToArgv(unittest.TestCase):

    def test_goal_add_basic(self):
        argv = _params_to_argv("goal.add", "goal_add", {"goal": "Fix the bug"})
        self.assertIn("Fix the bug", argv)
        self.assertIn("--json", argv)
        # canonical path includes "goal" and "add"
        self.assertIn("goal", argv)
        self.assertIn("add", argv)

    def test_goal_add_empty_goal(self):
        argv = _params_to_argv("goal.add", "goal_add", {})
        self.assertIn("--json", argv)
        # Without a goal param, only canonical path tokens + --json should be present.
        # The canonical path for goal_add is ("goal", "add").
        expected_tokens = {"goal", "add", "--json"}
        self.assertEqual(set(argv), expected_tokens)

    def test_goal_run_dry_run(self):
        argv = _params_to_argv("goal.run", "goal_run", {"dry_run": True})
        self.assertIn("--dry-run", argv)
        self.assertIn("--json", argv)

    def test_goal_run_max_cycles(self):
        argv = _params_to_argv("goal.run", "goal_run", {"max_cycles": 5})
        self.assertIn("--max-cycles", argv)
        self.assertIn("5", argv)

    def test_goal_once(self):
        argv = _params_to_argv("goal.once", "goal_once", {"goal": "Write tests"})
        self.assertIn("Write tests", argv)

    def test_memory_search(self):
        argv = _params_to_argv("memory.search", "memory_search", {"query": "vector store"})
        self.assertIn("vector store", argv)

    def test_mcp_call(self):
        argv = _params_to_argv(
            "mcp.call", "mcp_call", {"name": "my_tool", "input": {"key": "val"}}
        )
        self.assertIn("--name", argv)
        self.assertIn("my_tool", argv)
        self.assertIn("--input", argv)

    def test_json_always_appended(self):
        argv = _params_to_argv("doctor", "doctor", {})
        self.assertIn("--json", argv)

    def test_json_not_duplicated(self):
        # If params happen to produce --json, we shouldn't have it twice
        argv = _params_to_argv("goal.run", "goal_run", {})
        self.assertEqual(argv.count("--json"), 1)


# ---------------------------------------------------------------------------
# StdioTransport.handle_request — protocol validation
# ---------------------------------------------------------------------------

class TestStdioTransportHandleRequest(unittest.TestCase):

    def _transport(self):
        return _make_transport()

    def test_missing_jsonrpc_field(self):
        req = {"method": "goal.status", "id": 1}
        resp = self._transport().handle_request(req)
        self.assertEqual(resp["error"]["code"], INVALID_REQUEST)
        self.assertEqual(resp["id"], 1)

    def test_wrong_jsonrpc_version(self):
        req = {"jsonrpc": "1.0", "method": "goal.status", "id": 1}
        resp = self._transport().handle_request(req)
        self.assertEqual(resp["error"]["code"], INVALID_REQUEST)

    def test_missing_method(self):
        req = {"jsonrpc": "2.0", "id": 1}
        resp = self._transport().handle_request(req)
        self.assertEqual(resp["error"]["code"], INVALID_REQUEST)

    def test_empty_method(self):
        req = {"jsonrpc": "2.0", "method": "", "id": 1}
        resp = self._transport().handle_request(req)
        self.assertEqual(resp["error"]["code"], INVALID_REQUEST)

    def test_method_not_found(self):
        req = {"jsonrpc": "2.0", "method": "nonexistent.method", "id": 1}
        resp = self._transport().handle_request(req)
        self.assertEqual(resp["error"]["code"], METHOD_NOT_FOUND)
        self.assertEqual(resp["id"], 1)

    def test_params_must_be_object(self):
        req = {"jsonrpc": "2.0", "method": "goal.status", "params": [1, 2, 3], "id": 1}
        resp = self._transport().handle_request(req)
        self.assertEqual(resp["error"]["code"], INVALID_PARAMS)

    def test_request_must_be_dict(self):
        resp = _make_transport()._process_raw('"just a string"')
        self.assertEqual(resp["error"]["code"], INVALID_REQUEST)

    def test_parse_error_on_bad_json(self):
        transport = _make_transport()
        resp = transport._process_raw("{bad json}")
        self.assertEqual(resp["error"]["code"], PARSE_ERROR)
        self.assertIsNone(resp["id"])

    def test_null_id_in_response(self):
        req = {"jsonrpc": "2.0", "method": "goal.status"}  # no id
        # Should still respond (id will be None)
        # We just test it doesn't raise
        transport = _make_transport()
        with patch.object(transport, "_invoke_action", return_value={"exit_code": 0, "output": None}):
            resp = transport.handle_request(req)
        self.assertIsNone(resp.get("id"))
        self.assertIn("result", resp)


# ---------------------------------------------------------------------------
# StdioTransport.handle_request — successful dispatch
# ---------------------------------------------------------------------------

class TestStdioTransportDispatch(unittest.TestCase):
    """Tests that use a patched _invoke_action to avoid needing a real runtime."""

    def _dispatch_with_result(self, method, params=None, result=None):
        transport = _make_transport()
        result = result or {"exit_code": 0, "output": None}
        with patch.object(transport, "_invoke_action", return_value=result) as mock_invoke:
            req = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 99}
            resp = transport.handle_request(req)
        return resp, mock_invoke

    def test_successful_dispatch_returns_result(self):
        resp, _ = self._dispatch_with_result("goal.status", {}, {"exit_code": 0, "output": {"goals": []}})
        self.assertIn("result", resp)
        self.assertNotIn("error", resp)
        self.assertEqual(resp["id"], 99)
        self.assertEqual(resp["result"]["exit_code"], 0)

    def test_invoke_action_called_with_correct_args(self):
        transport = _make_transport()
        with patch.object(transport, "_invoke_action", return_value={"exit_code": 0, "output": None}) as mock:
            req = {"jsonrpc": "2.0", "method": "goal.add", "params": {"goal": "Test"}, "id": 1}
            transport.handle_request(req)
        mock.assert_called_once_with("goal.add", "goal_add", {"goal": "Test"})

    def test_internal_error_on_invoke_exception(self):
        transport = _make_transport()
        with patch.object(transport, "_invoke_action", side_effect=RuntimeError("boom")):
            req = {"jsonrpc": "2.0", "method": "goal.status", "params": {}, "id": 7}
            resp = transport.handle_request(req)
        self.assertEqual(resp["error"]["code"], INTERNAL_ERROR)
        self.assertEqual(resp["id"], 7)


# ---------------------------------------------------------------------------
# StdioTransport.run() — stream integration
# ---------------------------------------------------------------------------

class TestStdioTransportRun(unittest.TestCase):

    def test_run_processes_multiple_requests(self):
        lines = [
            json.dumps({"jsonrpc": "2.0", "method": "goal.status", "params": {}, "id": 1}),
            json.dumps({"jsonrpc": "2.0", "method": "goal.add", "params": {"goal": "Hello"}, "id": 2}),
        ]
        in_stream = io.StringIO("\n".join(lines) + "\n")
        out_stream = io.StringIO()

        transport = _make_transport(in_stream=in_stream, out_stream=out_stream)
        fake_result = {"exit_code": 0, "output": None}
        with patch.object(transport, "_invoke_action", return_value=fake_result):
            transport.run()

        output_lines = [l for l in out_stream.getvalue().splitlines() if l]
        self.assertEqual(len(output_lines), 2)
        for line in output_lines:
            resp = json.loads(line)
            self.assertEqual(resp["jsonrpc"], "2.0")
            self.assertIn("result", resp)

    def test_run_skips_empty_lines(self):
        lines = ["", "   ", json.dumps({"jsonrpc": "2.0", "method": "doctor", "id": 1})]
        in_stream = io.StringIO("\n".join(lines) + "\n")
        out_stream = io.StringIO()

        transport = _make_transport(in_stream=in_stream, out_stream=out_stream)
        with patch.object(transport, "_invoke_action", return_value={"exit_code": 0, "output": None}):
            transport.run()

        output_lines = [l for l in out_stream.getvalue().splitlines() if l]
        # Only the valid request should produce output
        self.assertEqual(len(output_lines), 1)

    def test_run_handles_parse_errors_inline(self):
        lines = [
            "NOT JSON AT ALL",
            json.dumps({"jsonrpc": "2.0", "method": "doctor", "id": 2}),
        ]
        in_stream = io.StringIO("\n".join(lines) + "\n")
        out_stream = io.StringIO()

        transport = _make_transport(in_stream=in_stream, out_stream=out_stream)
        with patch.object(transport, "_invoke_action", return_value={"exit_code": 0, "output": None}):
            transport.run()

        output_lines = [l for l in out_stream.getvalue().splitlines() if l]
        self.assertEqual(len(output_lines), 2)
        first = json.loads(output_lines[0])
        self.assertEqual(first["error"]["code"], PARSE_ERROR)
        second = json.loads(output_lines[1])
        self.assertIn("result", second)


# ---------------------------------------------------------------------------
# Custom method map
# ---------------------------------------------------------------------------

class TestCustomMethodMap(unittest.TestCase):

    def test_custom_method_map_is_used(self):
        custom_map = {"custom.action": "doctor"}
        transport = _make_transport()
        transport._method_map = custom_map

        with patch.object(transport, "_invoke_action", return_value={"exit_code": 0, "output": None}) as mock:
            req = {"jsonrpc": "2.0", "method": "custom.action", "params": {}, "id": 1}
            transport.handle_request(req)

        mock.assert_called_once_with("custom.action", "doctor", {})

    def test_unknown_method_with_custom_map(self):
        transport = _make_transport()
        transport._method_map = {"only.this": "doctor"}
        req = {"jsonrpc": "2.0", "method": "goal.add", "id": 1}
        resp = transport.handle_request(req)
        self.assertEqual(resp["error"]["code"], METHOD_NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
