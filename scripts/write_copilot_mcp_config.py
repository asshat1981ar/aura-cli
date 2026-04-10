#!/usr/bin/env python3
"""Generate a local GitHub Copilot MCP config for AURA's HTTP MCP servers.

By default this writes a safe config with ``${env:VAR}`` token placeholders instead
of embedding live secrets. The output defaults to ``.mcp.json`` because that file is
already ignored in this repository.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config_manager import ConfigManager


@dataclass(frozen=True)
class ServerSpec:
    config_name: str
    config_key: str
    output_name: str
    token_env: str | None
    port_envs: tuple[str, ...]


SERVER_SPECS: tuple[ServerSpec, ...] = (
    ServerSpec(
        config_name="dev_tools",
        config_key="aura-dev-tools",
        output_name="AURA dev tools",
        token_env="MCP_API_TOKEN",
        port_envs=("PORT", "MCP_SERVER_PORT"),
    ),
    ServerSpec(
        config_name="skills",
        config_key="aura-skills",
        output_name="AURA skills",
        token_env="MCP_API_TOKEN",
        port_envs=("MCP_SKILLS_PORT",),
    ),
    ServerSpec(
        config_name="control",
        config_key="aura-control",
        output_name="AURA control",
        token_env="MCP_CONTROL_TOKEN",
        port_envs=("MCP_CONTROL_PORT",),
    ),
    ServerSpec(
        config_name="agentic_loop",
        config_key="aura-agentic-loop",
        output_name="AURA agentic loop",
        token_env="AGENTIC_LOOP_TOKEN",
        port_envs=("AGENTIC_LOOP_PORT",),
    ),
    ServerSpec(
        config_name="copilot",
        config_key="aura-copilot",
        output_name="AURA copilot",
        token_env="COPILOT_MCP_TOKEN",
        port_envs=("COPILOT_MCP_PORT",),
    ),
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a local Copilot MCP config for AURA's HTTP MCP servers.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / ".mcp.json"),
        help="Path to the MCP config to write (default: %(default)s).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config file to read instead of aura.config.json/settings.json.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("AURA_COPILOT_MCP_HOST") or os.environ.get("AURA_MCP_HOST") or "127.0.0.1",
        help="Host name to use in generated MCP URLs (default: %(default)s).",
    )
    parser.add_argument(
        "--token-mode",
        choices=("placeholder", "omit"),
        default="placeholder",
        help="Write env placeholders for auth headers, or omit auth headers entirely.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated JSON to stdout after writing it.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing mcpServers content instead of merging unrelated entries.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def detect_config_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    for candidate in (ROOT / "aura.config.json", ROOT / "settings.json"):
        if candidate.exists():
            return candidate
    return ROOT / "aura.config.json"


def load_config(path: Path) -> ConfigManager:
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return ConfigManager(config_file=path)


def resolve_port(cfg: ConfigManager, spec: ServerSpec) -> int:
    for env_name in spec.port_envs:
        value = os.environ.get(env_name)
        if value:
            return int(value)
    return int(cfg.get_mcp_server_port(spec.config_name))


def placeholder_token(token_env: str) -> str:
    return f"${{env:{token_env}}}"


def build_headers(spec: ServerSpec, token_mode: str) -> Dict[str, str]:
    if not spec.token_env or token_mode == "omit":
        return {}
    return {"Authorization": f"Bearer {placeholder_token(spec.token_env)}"}


def build_server_entry(host: str, port: int, spec: ServerSpec, token_mode: str) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "type": "http",
        "url": f"http://{host}:{port}",
    }
    headers = build_headers(spec, token_mode)
    if headers:
        entry["headers"] = headers
    return entry


def load_existing_output(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def generate_config(
    *,
    cfg: ConfigManager,
    host: str,
    token_mode: str,
    replace: bool,
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {} if replace else dict(existing)
    mcp_servers = {} if replace else dict(output.get("mcpServers", {}))
    for spec in SERVER_SPECS:
        mcp_servers[spec.config_key] = build_server_entry(
            host=host,
            port=resolve_port(cfg, spec),
            spec=spec,
            token_mode=token_mode,
        )
    output["mcpServers"] = mcp_servers
    return output


def write_output(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = detect_config_path(args.config)
    cfg = load_config(config_path)
    output_path = Path(args.output).expanduser().resolve()
    existing = load_existing_output(output_path)
    payload = generate_config(
        cfg=cfg,
        host=args.host,
        token_mode=args.token_mode,
        replace=args.replace,
        existing=existing,
    )
    write_output(output_path, payload)
    print(f"Wrote {len(SERVER_SPECS)} AURA MCP servers to {output_path}")
    if args.stdout:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
