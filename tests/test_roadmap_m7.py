import pytest
from aura_cli.cli_main import main
from unittest.mock import patch, MagicMock
import io
from contextlib import redirect_stdout

def test_readiness_command_dispatch(monkeypatch):
    """Verify that 'readiness' command executes without error (M7-004)."""
    # Mock anyio.run to avoid actual async calls
    with patch("anyio.run", return_value=[]), \
         patch("core.mcp_health.get_health_summary", return_value={"healthy_count": 0, "total_servers": 0, "all_healthy": True}):
        
        f = io.StringIO()
        with redirect_stdout(f):
            # Run readiness command
            # Set AURA_SKIP_CHDIR=1 to avoid path issues in tests
            monkeypatch.setenv("AURA_SKIP_CHDIR", "1")
            exit_code = main(argv=["readiness"])
            
        output = f.getvalue()
        assert exit_code == 0
        assert "AURA Readiness Check (V2)" in output
        assert "Async Orchestrator: ENABLED" in output
