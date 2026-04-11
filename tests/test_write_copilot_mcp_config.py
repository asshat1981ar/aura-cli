import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "write_copilot_mcp_config.py"


def _run_script(*args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("AURA_SKIP_CHDIR", "1")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_script_writes_placeholder_headers_and_default_ports(tmp_path: Path):
    output_path = tmp_path / "generated.mcp.json"

    result = _run_script("--output", str(output_path))

    assert result.returncode == 0, result.stderr
    assert "Wrote 7 MCP servers" in result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    servers = payload["mcpServers"]
    assert servers["aura-dev-tools"]["url"] == "http://127.0.0.1:8001"
    assert servers["aura-skills"]["url"] == "http://127.0.0.1:8002"
    assert servers["aura-control"]["url"] == "http://127.0.0.1:8003"
    assert servers["aura-agentic-loop"]["url"] == "http://127.0.0.1:8006"
    assert servers["aura-copilot"]["url"] == "http://127.0.0.1:8007"
    assert servers["aura-sadd"]["url"] == "http://127.0.0.1:8020"
    assert servers["aura-dev-tools"]["headers"]["Authorization"] == "Bearer ${env:AGENT_API_TOKEN}"
    assert servers["aura-copilot"]["headers"]["Authorization"] == "Bearer ${env:COPILOT_MCP_TOKEN}"
    assert servers["aura-sadd"]["headers"]["Authorization"] == "Bearer ${env:SADD_MCP_TOKEN}"
    assert servers["playwright"]["type"] == "stdio"
    assert servers["playwright"]["command"] == "npx"


def test_script_respects_env_port_overrides_and_preserves_unmanaged_servers(tmp_path: Path):
    output_path = tmp_path / "generated.mcp.json"
    output_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "existing-server": {
                        "type": "stdio",
                        "command": "echo",
                        "args": ["ok"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_script(
        "--output",
        str(output_path),
        "--host",
        "localhost",
        extra_env={
            "PORT": "9101",
            "MCP_SKILLS_PORT": "9102",
            "MCP_CONTROL_PORT": "9103",
            "AGENTIC_LOOP_PORT": "9106",
            "COPILOT_MCP_PORT": "9107",
            "SADD_MCP_PORT": "9120",
        },
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    servers = payload["mcpServers"]
    assert servers["existing-server"]["command"] == "echo"
    assert servers["aura-dev-tools"]["url"] == "http://localhost:9101"
    assert servers["aura-skills"]["url"] == "http://localhost:9102"
    assert servers["aura-control"]["url"] == "http://localhost:9103"
    assert servers["aura-agentic-loop"]["url"] == "http://localhost:9106"
    assert servers["aura-copilot"]["url"] == "http://localhost:9107"
    assert servers["aura-sadd"]["url"] == "http://localhost:9120"
    assert servers["playwright"]["args"] == ["@playwright/mcp@latest"]


def test_script_can_omit_auth_headers(tmp_path: Path):
    output_path = tmp_path / "generated.mcp.json"

    result = _run_script("--output", str(output_path), "--token-mode", "omit")

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    for server in payload["mcpServers"].values():
        assert "headers" not in server
