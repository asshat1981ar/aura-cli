from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROTOCOL_VERSION = "2024-11-05"
DEFAULT_INIT_TIMEOUT_SECONDS = 3.0
DEFAULT_LIST_TIMEOUT_SECONDS = 5.0
DEFAULT_CALL_TIMEOUT_SECONDS = 20.0
ENV_PATTERN = re.compile(r"\$\{(?:(env):)?([A-Za-z_][A-Za-z0-9_]*)\}")
ROOT = Path(__file__).resolve().parents[1]


class MCPCLIError(RuntimeError):
    pass


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    config: dict[str, Any]

    @property
    def transport(self) -> str:
        return str(self.config.get("type", "stdio"))


def _resolve_config_path(explicit_path: str | None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    if os.environ.get("AURA_MCP_CONFIG"):
        candidates.append(Path(os.environ["AURA_MCP_CONFIG"]).expanduser())
    candidates.extend(
        (
            Path.cwd() / ".mcp.json",
            ROOT / ".mcp.json",
        )
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: os.environ.get(match.group(2), ""), value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MCPCLIError(f"MCP config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_server_specs(path: Path) -> dict[str, MCPServerSpec]:
    payload = _load_config(path)
    servers = payload.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise MCPCLIError(f"Invalid MCP config: expected mcpServers object in {path}")
    return {name: MCPServerSpec(name=name, config=config) for name, config in servers.items()}


def _json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _resolve_timeout(config: dict[str, Any], env_var: str, config_key: str, default: float) -> float:
    raw_value = os.environ.get(env_var, config.get(config_key, default))
    try:
        timeout = float(raw_value)
    except (TypeError, ValueError):
        return default
    return timeout if timeout > 0 else default


class StdioMCPClient:
    def __init__(self, spec: MCPServerSpec):
        config = _expand_env_vars(spec.config)
        self.spec = spec
        self.command = config.get("command")
        self.args = list(config.get("args", []))
        self.env = {**os.environ, **config.get("env", {})}
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._req_id = 0
        self._pending: dict[int, threading.Event] = {}
        self._responses: dict[int, dict[str, Any]] = {}
        self._stderr_lines: list[str] = []
        self._tools_cache: list[dict[str, Any]] | None = None
        self._initialize_timeout = _resolve_timeout(
            config,
            "AURA_MCP_INIT_TIMEOUT_SECONDS",
            "initialize_timeout_seconds",
            DEFAULT_INIT_TIMEOUT_SECONDS,
        )
        self._list_timeout = _resolve_timeout(
            config,
            "AURA_MCP_LIST_TIMEOUT_SECONDS",
            "list_timeout_seconds",
            DEFAULT_LIST_TIMEOUT_SECONDS,
        )
        self._call_timeout = _resolve_timeout(
            config,
            "AURA_MCP_CALL_TIMEOUT_SECONDS",
            "call_timeout_seconds",
            DEFAULT_CALL_TIMEOUT_SECONDS,
        )

    def __enter__(self) -> "StdioMCPClient":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            return
        if not self.command:
            raise MCPCLIError(f"Server '{self.spec.name}' is missing a command")
        try:
            self._proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=self.env,
            )
        except OSError as exc:
            raise MCPCLIError(f"Failed to start server '{self.spec.name}': {exc}") from exc

        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._initialize()

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def _initialize(self) -> None:
        response = self._rpc(
            "initialize",
            {
                "protocolVersion": DEFAULT_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "aura-mcp-cli", "version": "0.1.0"},
            },
            timeout=self._initialize_timeout,
        )
        if "error" in response:
            raise MCPCLIError(f"Server '{self.spec.name}' rejected initialize: {response['error']}")
        self._notify("notifications/initialized", {})

    def _read_stdout(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            req_id = payload.get("id")
            if req_id is None:
                continue
            event = self._pending.get(req_id)
            if event is None:
                continue
            self._responses[req_id] = payload
            event.set()

    def _read_stderr(self) -> None:
        assert self._proc is not None
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            text = line.strip()
            if text:
                self._stderr_lines.append(text)

    def _rpc(self, method: str, params: dict[str, Any], timeout: float = 20.0) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None:
            raise MCPCLIError(f"Server '{self.spec.name}' is not running")
        with self._lock:
            self._req_id += 1
            req_id = self._req_id
            event = threading.Event()
            self._pending[req_id] = event
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise MCPCLIError(f"Server '{self.spec.name}' closed its stdin") from exc
        if not event.wait(timeout):
            stderr = f" stderr: {' | '.join(self._stderr_lines[-5:])}" if self._stderr_lines else ""
            self._pending.pop(req_id, None)
            raise MCPCLIError(f"Timed out waiting for '{method}' from '{self.spec.name}'.{stderr}")
        self._pending.pop(req_id, None)
        return self._responses.pop(req_id)

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError:
            return

    def list_tools(self) -> list[dict[str, Any]]:
        if self._tools_cache is not None:
            return self._tools_cache
        response = self._rpc("tools/list", {}, timeout=self._list_timeout)
        if "error" in response:
            raise MCPCLIError(f"tools/list failed for '{self.spec.name}': {response['error']}")
        self._tools_cache = list(response.get("result", {}).get("tools", []))
        return self._tools_cache

    def get_tool(self, tool_name: str) -> dict[str, Any]:
        for tool in self.list_tools():
            if tool.get("name") == tool_name:
                return tool
        raise MCPCLIError(f"Tool '{tool_name}' not found on server '{self.spec.name}'")

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = self._rpc("tools/call", {"name": tool_name, "arguments": arguments}, timeout=self._call_timeout)
        if "error" in response:
            raise MCPCLIError(f"tools/call failed for '{self.spec.name}/{tool_name}': {response['error']}")
        return response.get("result", {})


class HttpMCPClient:
    def __init__(self, spec: MCPServerSpec):
        config = _expand_env_vars(spec.config)
        self.spec = spec
        self.base_url = str(config.get("url", "")).rstrip("/")
        if not self.base_url:
            raise MCPCLIError(f"Server '{spec.name}' is missing a url")
        self.headers = dict(config.get("headers", {}))
        self._tools_cache: list[dict[str, Any]] | None = None
        self._jsonrpc_initialized = False
        self._initialize_timeout = _resolve_timeout(
            config,
            "AURA_MCP_INIT_TIMEOUT_SECONDS",
            "initialize_timeout_seconds",
            DEFAULT_INIT_TIMEOUT_SECONDS,
        )
        self._list_timeout = _resolve_timeout(
            config,
            "AURA_MCP_LIST_TIMEOUT_SECONDS",
            "list_timeout_seconds",
            DEFAULT_LIST_TIMEOUT_SECONDS,
        )
        self._call_timeout = _resolve_timeout(
            config,
            "AURA_MCP_CALL_TIMEOUT_SECONDS",
            "call_timeout_seconds",
            DEFAULT_CALL_TIMEOUT_SECONDS,
        )

    def __enter__(self) -> "HttpMCPClient":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def _request(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
    ) -> Any:
        headers = {"Accept": "application/json", **self.headers, **(extra_headers or {})}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode()
            try:
                payload = json.loads(body_text)
            except json.JSONDecodeError:
                payload = {"error": body_text or str(exc)}
            raise MCPCLIError(f"HTTP {exc.code} from '{self.spec.name}': {_json_dump(payload)}") from exc
        except urllib.error.URLError as exc:
            raise MCPCLIError(f"Failed to reach server '{self.spec.name}': {exc.reason}") from exc

    def _request_aura(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
    ) -> Any:
        return self._request(method, f"{self.base_url}{path}", payload, timeout=timeout)

    def _request_jsonrpc(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        if not self._jsonrpc_initialized:
            response = self._request(
                "POST",
                self.base_url,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": DEFAULT_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "aura-mcp-cli", "version": "0.1.0"},
                    },
                },
                timeout=self._initialize_timeout,
            )
            if "error" in response:
                raise MCPCLIError(f"Server '{self.spec.name}' rejected initialize: {response['error']}")
            self._jsonrpc_initialized = True
        return self._request(
            "POST",
            self.base_url,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": method,
                "params": params,
            },
            timeout=timeout,
        )

    def _aura_tool_list(self) -> list[dict[str, Any]]:
        payload = self._request_aura("GET", "/tools", timeout=self._list_timeout)
        if isinstance(payload, dict):
            return list(payload.get("tools", []))
        if isinstance(payload, list):
            return payload
        raise MCPCLIError(f"Unexpected tool list response from '{self.spec.name}'")

    def list_tools(self) -> list[dict[str, Any]]:
        if self._tools_cache is not None:
            return self._tools_cache
        try:
            self._tools_cache = self._aura_tool_list()
            return self._tools_cache
        except MCPCLIError:
            response = self._request_jsonrpc("tools/list", {}, timeout=self._list_timeout)
            if "error" in response:
                raise MCPCLIError(f"tools/list failed for '{self.spec.name}': {response['error']}")
            self._tools_cache = list(response.get("result", {}).get("tools", []))
            return self._tools_cache

    def get_tool(self, tool_name: str) -> dict[str, Any]:
        try:
            payload = self._request_aura(
                "GET",
                f"/tool/{urllib.parse.quote(tool_name, safe='')}",
                timeout=self._list_timeout,
            )
            if isinstance(payload, dict):
                return payload
        except MCPCLIError:
            pass
        for tool in self.list_tools():
            if tool.get("name") == tool_name:
                return tool
        raise MCPCLIError(f"Tool '{tool_name}' not found on server '{self.spec.name}'")

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        try:
            return self._request_aura(
                "POST",
                "/call",
                {"tool_name": tool_name, "args": arguments},
                timeout=self._call_timeout,
            )
        except MCPCLIError:
            response = self._request_jsonrpc(
                "tools/call",
                {"name": tool_name, "arguments": arguments},
                timeout=self._call_timeout,
            )
            if "error" in response:
                raise MCPCLIError(f"tools/call failed for '{self.spec.name}/{tool_name}': {response['error']}")
            return response.get("result", {})


def _create_client(spec: MCPServerSpec) -> StdioMCPClient | HttpMCPClient:
    if spec.transport == "http":
        return HttpMCPClient(spec)
    return StdioMCPClient(spec)


def _format_tool_line(server_name: str, tool: dict[str, Any], include_description: bool) -> str:
    name = str(tool.get("name", "<unknown>"))
    description = str(tool.get("description", "")).strip()
    if include_description and description:
        return f"{server_name}/{name} - {description}"
    return f"{server_name}/{name}"


def _list_servers(server_specs: dict[str, MCPServerSpec], include_description: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(server_specs):
        spec = server_specs[name]
        row: dict[str, Any] = {"server": name, "transport": spec.transport}
        try:
            with _create_client(spec) as client:
                tools = client.list_tools()
            row["tools"] = [
                {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "inputSchema": tool.get("inputSchema"),
                }
                for tool in tools
            ]
        except MCPCLIError as exc:
            row["error"] = str(exc)
            row["tools"] = []
        rows.append(row)
    if not include_description:
        for row in rows:
            for tool in row["tools"]:
                tool.pop("description", None)
    return rows


def _print_default_listing(rows: list[dict[str, Any]], raw: bool, include_description: bool) -> None:
    if raw:
        for row in rows:
            tools = row.get("tools", [])
            if not tools:
                print(row["server"])
                continue
            for tool in tools:
                print(_format_tool_line(row["server"], tool, include_description))
        return
    for row in rows:
        header = f"{row['server']} ({row['transport']})"
        if row.get("error"):
            print(f"{header} [error: {row['error']}]")
            continue
        print(header)
        tools = row.get("tools", [])
        if not tools:
            print("  (no tools)")
            continue
        for tool in tools:
            print(f"  - {_format_tool_line(row['server'], tool, include_description)}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repo-local MCP CLI for AURA")
    parser.add_argument("target", nargs="?", help="Optional server, server/tool, or 'grep' target.")
    parser.add_argument("payload", nargs="?", help="JSON arguments for tool calls or grep pattern.")
    parser.add_argument("-c", "--config", help="Path to MCP config (default: ./.mcp.json or $AURA_MCP_CONFIG).")
    parser.add_argument("-j", "--json", action="store_true", help="Print JSON output.")
    parser.add_argument("-r", "--raw", action="store_true", help="Print raw text when possible.")
    parser.add_argument("-d", "--descriptions", action="store_true", help="Include tool descriptions.")
    return parser.parse_args(argv)


def _split_target(target: str) -> tuple[str, str | None]:
    if "/" not in target:
        return target, None
    server_name, tool_name = target.split("/", 1)
    return server_name, tool_name


def _handle_grep(pattern: str, server_specs: dict[str, MCPServerSpec], args: argparse.Namespace) -> int:
    matches: list[dict[str, Any]] = []
    rows = _list_servers(server_specs, include_description=True)
    for row in rows:
        if fnmatch.fnmatch(row["server"], pattern):
            matches.append(row)
        for tool in row.get("tools", []):
            name = f"{row['server']}/{tool.get('name', '')}"
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(str(tool.get("name", "")), pattern):
                matches.append({"server": row["server"], "tool": tool})
    if args.json:
        print(_json_dump({"pattern": pattern, "matches": matches}))
    else:
        if not matches:
            return 1
        for match in matches:
            if "tool" in match:
                print(_format_tool_line(match["server"], match["tool"], args.descriptions))
            else:
                print(match["server"])
    return 0 if matches else 1


def _handle_server(server_name: str, server_specs: dict[str, MCPServerSpec], args: argparse.Namespace) -> int:
    spec = server_specs.get(server_name)
    if spec is None:
        raise MCPCLIError(f"Unknown MCP server '{server_name}'")
    with _create_client(spec) as client:
        tools = client.list_tools()
    payload = {
        "server": server_name,
        "transport": spec.transport,
        "config": spec.config if args.descriptions else {"type": spec.transport},
        "tools": tools,
    }
    if not args.descriptions:
        payload["tools"] = [
            {
                "name": tool.get("name"),
                "inputSchema": tool.get("inputSchema"),
            }
            for tool in tools
        ]
    if args.json:
        print(_json_dump(payload))
    elif args.raw:
        for tool in tools:
            print(_format_tool_line(server_name, tool, args.descriptions))
    else:
        print(_json_dump(payload))
    return 0


def _extract_raw_text(result: Any) -> str | None:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            texts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
            if texts:
                return "\n".join(texts)
        if "result" in result and isinstance(result["result"], str):
            return result["result"]
    return None


def _handle_tool(server_name: str, tool_name: str, server_specs: dict[str, MCPServerSpec], args: argparse.Namespace) -> int:
    spec = server_specs.get(server_name)
    if spec is None:
        raise MCPCLIError(f"Unknown MCP server '{server_name}'")
    with _create_client(spec) as client:
        if args.payload is None:
            payload = client.get_tool(tool_name)
        else:
            try:
                tool_args = json.loads(args.payload)
            except json.JSONDecodeError as exc:
                raise MCPCLIError(f"Invalid JSON payload for '{server_name}/{tool_name}': {exc}") from exc
            payload = client.call_tool(tool_name, tool_args)
    if args.raw:
        text = _extract_raw_text(payload)
        if text is not None:
            print(text)
            return 0
    if args.json or not args.raw:
        print(_json_dump(payload))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = _resolve_config_path(args.config)
    try:
        server_specs = _load_server_specs(config_path)
        if args.target is None:
            rows = _list_servers(server_specs, include_description=args.descriptions)
            if args.json:
                print(_json_dump({"config_path": str(config_path), "servers": rows}))
            else:
                _print_default_listing(rows, args.raw, args.descriptions)
            return 0

        if args.target == "grep":
            if not args.payload:
                raise MCPCLIError("grep requires a glob pattern, for example: aura-mcp-cli grep '*file*'")
            return _handle_grep(args.payload, server_specs, args)

        server_name, tool_name = _split_target(args.target)
        if tool_name is None:
            return _handle_server(server_name, server_specs, args)
        return _handle_tool(server_name, tool_name, server_specs, args)
    except MCPCLIError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
