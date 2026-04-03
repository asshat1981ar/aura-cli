"""Unit tests for core/beads_contract.py — TypedDicts, dataclass, constants."""
from core.beads_contract import (
    BEADS_SCHEMA_VERSION,
    BeadsBridgeConfig,
    BeadsDecision,
    BeadsInput,
    BeadsResult,
)


class TestBeadsSchemaVersion:
    def test_schema_version_is_integer(self):
        assert isinstance(BEADS_SCHEMA_VERSION, int)

    def test_schema_version_value(self):
        assert BEADS_SCHEMA_VERSION == 1


class TestBeadsBridgeConfig:
    def test_default_values(self):
        cfg = BeadsBridgeConfig(command=("bd", "run"))
        assert cfg.timeout_seconds == 20.0
        assert cfg.enabled is True
        assert cfg.required is True
        assert cfg.persist_artifacts is True
        assert cfg.scope == "goal_run"
        assert cfg.env == {}

    def test_custom_command_stored(self):
        cfg = BeadsBridgeConfig(command=("python", "run.py", "--dry-run"))
        assert cfg.command == ("python", "run.py", "--dry-run")

    def test_custom_timeout(self):
        cfg = BeadsBridgeConfig(command=("bd",), timeout_seconds=5.0)
        assert cfg.timeout_seconds == 5.0

    def test_frozen_prevents_mutation(self):
        import pytest
        cfg = BeadsBridgeConfig(command=("bd",))
        with pytest.raises((AttributeError, TypeError)):
            cfg.enabled = False  # type: ignore[misc]

    def test_custom_env_dict(self):
        cfg = BeadsBridgeConfig(command=("bd",), env={"MY_KEY": "value"})
        assert cfg.env == {"MY_KEY": "value"}

    def test_env_instances_are_independent(self):
        cfg1 = BeadsBridgeConfig(command=("bd",))
        cfg2 = BeadsBridgeConfig(command=("bd",))
        assert cfg1.env is not cfg2.env


class TestBeadsTypedDicts:
    def test_beads_input_can_be_constructed(self):
        inp: BeadsInput = {
            "schema_version": 1,
            "goal": "Add tests",
            "goal_type": None,
            "runtime_mode": "autonomous",
            "project_root": "/project",
            "queue_summary": {},
            "active_context": {},
            "prd_context": None,
            "conductor_track": None,
        }
        assert inp["goal"] == "Add tests"
        assert inp["schema_version"] == 1

    def test_beads_result_ok_field(self):
        result: BeadsResult = {
            "schema_version": 1,
            "ok": True,
            "status": "ok",
            "decision": None,
            "error": None,
            "stderr": None,
            "duration_ms": 150,
        }
        assert result["ok"] is True
        assert result["duration_ms"] == 150
