import pytest
import os
from pathlib import Path
from agents.sandbox import SandboxAgent

def test_sandbox_run_code_with_local_import(tmp_path):
    # Create a dummy module in a temporary project root
    project_root = tmp_path / "my_project"
    project_root.mkdir()
    (project_root / "utils.py").write_text("def get_secret(): return 42")
    (project_root / "__init__.py").touch()

    agent = SandboxAgent(brain=None)
    
    code = "from utils import get_secret\nprint(get_secret())"
    
    # Run without project_root should fail (unless utils is in PYTHONPATH)
    res_fail = agent.run_code(code)
    assert not res_fail.success
    assert "ModuleNotFoundError" in res_fail.stderr

    # Run with project_root should pass
    res_pass = agent.run_code(code, project_root=str(project_root))
    assert res_pass.success
    assert res_pass.stdout.strip() == "42"
