import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "run_aura.sh"


def _run_wrapper(*args: str, echo: bool = False, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("AURA_SKIP_CHDIR", "1")
    if echo:
        env["AURA_WRAPPER_ECHO"] = "1"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_wrapper_help_mentions_aliases_and_passthrough():
    result = _run_wrapper("--help")

    assert result.returncode == 0
    assert "run                Forward to `python3 main.py goal run`" in result.stdout
    assert "./run_aura.sh goal run --dry-run" in result.stdout
    assert result.stderr == ""


def test_wrapper_run_alias_maps_to_goal_run():
    result = _run_wrapper("run", "--dry-run", echo=True)

    assert result.returncode == 0
    assert result.stdout.splitlines()[-3:] == ["goal", "run", "--dry-run"]


def test_wrapper_leading_flags_default_to_goal_run():
    result = _run_wrapper("--dry-run", "--max-cycles", "1", echo=True)

    assert result.returncode == 0
    assert result.stdout.splitlines()[-5:] == ["goal", "run", "--dry-run", "--max-cycles", "1"]


def test_wrapper_preserves_canonical_commands():
    result = _run_wrapper("contract-report", "--check", echo=True)

    assert result.returncode == 0
    assert result.stdout.splitlines()[-2:] == ["contract-report", "--check"]


def test_wrapper_add_uses_real_goal_add_path_without_heavy_runtime_startup():
    with tempfile.TemporaryDirectory() as tmpdir:
        queue_path = Path(tmpdir) / "goal_queue.json"
        result = _run_wrapper(
            "add",
            "Wrapper smoke goal",
            extra_env={"AURA_GOAL_QUEUE_PATH": str(queue_path)},
        )

        assert result.returncode == 0
        assert json.loads(queue_path.read_text(encoding="utf-8")) == ["Wrapper smoke goal"]
        assert "vector_store_initialized" not in result.stderr
        assert "background_sync_started" not in result.stderr


def test_wrapper_status_json_stays_machine_readable_and_lightweight():
    with tempfile.TemporaryDirectory() as tmpdir:
        queue_path = Path(tmpdir) / "goal_queue.json"
        queue_path.write_text(json.dumps(["Wrapper queued goal"]), encoding="utf-8")

        result = _run_wrapper(
            "status",
            "--json",
            extra_env={"AURA_GOAL_QUEUE_PATH": str(queue_path)},
        )

        assert result.returncode == 0
        assert json.loads(result.stdout)["queue"] == ["Wrapper queued goal"]
        assert "vector_store_initialized" not in result.stderr
        assert "background_sync_started" not in result.stderr
