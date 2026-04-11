from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / ".vscode" / "mcp.json.example"
LSP_CONFIG = ROOT / ".github" / "lsp.json"
SCRIPT = ROOT / "scripts" / "configure_copilot_mcp.sh"


def test_mcp_example_is_valid_and_secret_free() -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    assert set(payload["mcpServers"]) == {
        "aura-dev-tools",
        "aura-skills",
        "aura-control",
        "aura-agentic-loop",
        "aura-copilot",
        "n8n-mcp",
    }
    serialized = json.dumps(payload)
    assert "ghp_" not in serialized
    assert "Authorization" in serialized
    assert "${env:MCP_API_TOKEN}" in serialized
    assert "${env:MCP_CONTROL_TOKEN}" in serialized
    assert "${env:AGENTIC_LOOP_TOKEN}" in serialized
    assert "${env:COPILOT_MCP_TOKEN}" in serialized
    assert "${env:N8N_MCP_TOKEN}" in serialized


def test_repo_lsp_config_includes_ruff_and_pyright_servers() -> None:
    payload = json.loads(LSP_CONFIG.read_text(encoding="utf-8"))
    ruff_lsp = payload["lspServers"]["ruff"]
    pyright_lsp = payload["lspServers"]["pyright"]

    assert ruff_lsp["command"] == "ruff"
    assert ruff_lsp["args"] == ["server"]
    assert ruff_lsp["fileExtensions"][".py"] == "python"
    assert pyright_lsp["command"] == "pyright-langserver"
    assert pyright_lsp["args"] == ["--stdio"]
    assert pyright_lsp["fileExtensions"][".py"] == "python"


def test_configure_copilot_mcp_writes_expected_config() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "mcp.json"
        env = dict(os.environ)
        env.update(
            {
                "COPILOT_MCP_CONFIG_PATH": str(out_path),
                "COPILOT_MCP_BASE_URL": "http://127.0.0.1",
            }
        )

        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        assert result.returncode == 0, result.stderr

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["mcpServers"]["aura-dev-tools"]["url"] == "http://127.0.0.1:8001"
        assert payload["mcpServers"]["aura-skills"]["url"] == "http://127.0.0.1:8002"
        assert payload["mcpServers"]["aura-control"]["url"] == "http://127.0.0.1:8003"
        assert payload["mcpServers"]["aura-agentic-loop"]["url"] == "http://127.0.0.1:8006"
        assert payload["mcpServers"]["aura-copilot"]["url"] == "http://127.0.0.1:8007"
        assert payload["mcpServers"]["aura-dev-tools"]["headers"]["Authorization"] == "Bearer ${env:MCP_API_TOKEN}"
        assert payload["mcpServers"]["aura-skills"]["headers"]["Authorization"] == "Bearer ${env:MCP_API_TOKEN}"
        assert payload["mcpServers"]["aura-control"]["headers"]["Authorization"] == "Bearer ${env:MCP_CONTROL_TOKEN}"
        assert payload["mcpServers"]["aura-agentic-loop"]["headers"]["Authorization"] == "Bearer ${env:AGENTIC_LOOP_TOKEN}"
        assert payload["mcpServers"]["aura-copilot"]["headers"]["Authorization"] == "Bearer ${env:COPILOT_MCP_TOKEN}"
