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
        raw = json.loads(queue_path.read_text(encoding="utf-8"))
        # Support both old list format and new {queue, in_flight} format
        goals = raw if isinstance(raw, list) else raw.get("queue", raw)
        assert goals == ["Wrapper smoke goal"]
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
        status_data = json.loads(result.stdout)
        assert status_data["queue"]["pending"][0]["goal"] == "Wrapper queued goal"
        assert "vector_store_initialized" not in result.stderr
        assert "background_sync_started" not in result.stderr


def test_wrapper_can_require_local_model_health_before_runtime_commands():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config_path = tmp / "aura.config.json"
        log_path = tmp / "wrapper.log"
        fake_python = tmp / "fake_python.sh"

        config_path.write_text(
            json.dumps(
                {
                    "local_model_profiles": {
                        "android_coder": {
                            "provider": "openai_compatible",
                            "base_url": "http://127.0.0.1:8080/v1",
                        },
                        "android_planner": {
                            "provider": "openai_compatible",
                            "base_url": "http://127.0.0.1:8081/v1",
                        },
                        "android_embeddings": {
                            "provider": "openai_compatible",
                            "base_url": "http://127.0.0.1:8082/v1",
                        },
                    },
                    "local_model_routing": {
                        "code_generation": "android_coder",
                        "planning": "android_planner",
                        "embedding": "android_embeddings",
                    },
                    "semantic_memory": {
                        "embedding_model": "local_profile:android_embeddings",
                    },
                }
            ),
            encoding="utf-8",
        )
        fake_python.write_text(
            '#!/usr/bin/env bash\nscript="$1"\nshift\nprintf \'%s %s\\n\' "$script" "$*" >> "$AURA_WRAPPER_LOG"\nexit 0\n',
            encoding="utf-8",
        )
        fake_python.chmod(0o700)

        result = _run_wrapper(
            "run",
            "--dry-run",
            extra_env={
                "AURA_REQUIRE_LOCAL_MODEL_HEALTH": "1",
                "AURA_ANDROID_CONFIG_PATH": str(config_path),
                "AURA_PYTHON_BIN": str(fake_python),
                "AURA_WRAPPER_LOG": str(log_path),
            },
        )

        assert result.returncode == 0
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert "scripts/check_android_local_models.py --config " in lines[0]
        assert "main.py goal run --dry-run" in lines[1]
