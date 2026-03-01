import subprocess
from pathlib import Path
from unittest.mock import patch

from agents.verifier import VerifierAgent


def test_verifier_prefers_targeted_tests_for_repo_wide_default(tmp_path: Path):
    project_root = tmp_path
    (project_root / "tests").mkdir()
    (project_root / "tests" / "test_run_aura_wrapper.py").write_text("def test_ok():\n    pass\n", encoding="utf-8")
    (project_root / "run_aura.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    agent = VerifierAgent(timeout=5)
    completed = subprocess.CompletedProcess(
        args=["python3", "-m", "pytest", "-q", "tests/test_run_aura_wrapper.py"],
        returncode=0,
        stdout="1 passed\n",
        stderr="",
    )

    with patch("agents.verifier.subprocess.run", return_value=completed) as mock_run, \
         patch.dict("os.environ", {}, clear=True):
        result = agent.run(
            {
                "project_root": str(project_root),
                "change_set": {"changes": [{"file_path": "run_aura.sh"}]},
                "tests": ["python3 -m pytest -q"],
            }
        )

    assert result["status"] == "pass"
    assert mock_run.call_args.kwargs["env"]["AURA_SKIP_CHDIR"] == "1"
    assert mock_run.call_args.args[0] == [
        "python3",
        "-m",
        "pytest",
        "-q",
        "tests/test_run_aura_wrapper.py",
    ]


def test_verifier_keeps_explicit_targeted_command(tmp_path: Path):
    project_root = tmp_path
    (project_root / "tests").mkdir()
    (project_root / "tests" / "test_cli_options.py").write_text("def test_ok():\n    pass\n", encoding="utf-8")

    agent = VerifierAgent(timeout=5)
    completed = subprocess.CompletedProcess(
        args=["python3", "-m", "pytest", "-q", "tests/test_cli_options.py"],
        returncode=0,
        stdout="1 passed\n",
        stderr="",
    )

    with patch("agents.verifier.subprocess.run", return_value=completed) as mock_run:
        result = agent.run(
            {
                "project_root": str(project_root),
                "tests": ["python3 -m pytest -q tests/test_cli_options.py"],
            }
        )

    assert result["status"] == "pass"
    assert mock_run.call_args.args[0] == [
        "python3",
        "-m",
        "pytest",
        "-q",
        "tests/test_cli_options.py",
    ]


def test_verifier_timeout_logs_command(tmp_path: Path):
    agent = VerifierAgent(timeout=3)

    with patch(
        "agents.verifier.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["pytest", "-q"], timeout=3),
    ):
        result = agent.run({"project_root": str(tmp_path), "tests": ["pytest -q"]})

    assert result["status"] == "fail"
    assert result["failures"] == ["pytest_timeout"]
    assert "timeout after 3s while running: pytest -q" == result["logs"]
