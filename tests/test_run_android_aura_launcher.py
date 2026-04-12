import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_android_aura.sh"


def _run_launcher(*args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_launcher_defaults_to_watch_and_enables_health_gate():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        wrapper = tmp / "fake_wrapper.sh"
        log = tmp / "launcher.log"

        wrapper.write_text(
            '#!/usr/bin/env bash\nprintf \'mode=%s\\n\' "$1" > "$AURA_LAUNCHER_LOG"\nprintf \'gate=%s\\n\' "$AURA_REQUIRE_LOCAL_MODEL_HEALTH" >> "$AURA_LAUNCHER_LOG"\nprintf \'rest=%s\\n\' "$*" >> "$AURA_LAUNCHER_LOG"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o700)

        result = _run_launcher(
            extra_env={
                "AURA_ANDROID_WRAPPER": str(wrapper),
                "AURA_LAUNCHER_LOG": str(log),
            }
        )

        assert result.returncode == 0
        contents = log.read_text(encoding="utf-8")
        assert "mode=watch" in contents
        assert "gate=1" in contents
        assert "rest=watch" in contents


def test_launcher_forwards_studio_flags():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        wrapper = tmp / "fake_wrapper.sh"
        log = tmp / "launcher.log"

        wrapper.write_text(
            '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$AURA_LAUNCHER_LOG"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o700)

        result = _run_launcher(
            "studio",
            "--autonomous",
            extra_env={
                "AURA_ANDROID_WRAPPER": str(wrapper),
                "AURA_LAUNCHER_LOG": str(log),
            },
        )

        assert result.returncode == 0
        assert log.read_text(encoding="utf-8").splitlines() == ["studio", "--autonomous"]


def test_launcher_rejects_invalid_mode():
    result = _run_launcher("goal")

    assert result.returncode == 2
    assert "expected `watch` or `studio`" in result.stderr
