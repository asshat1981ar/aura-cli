"""E5: Tests for BEADS bridge retry, offline queue, and conflict detection."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.beads_bridge import BeadsBridge, _result, _TRANSIENT_ERRORS
from core.beads_contract import BEADS_SCHEMA_VERSION, BeadsBridgeConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_bridge(tmp_path: Path, **kwargs) -> BeadsBridge:
    defaults = dict(
        command=("echo", "test"),
        timeout_seconds=5.0,
        enabled=True,
        required=False,
        persist_artifacts=True,
    )
    defaults.update(kwargs)
    cfg = BeadsBridgeConfig(**defaults)
    return BeadsBridge(cfg, project_root=tmp_path)


def _ok_result(**overrides) -> dict:
    base = _result(ok=True, status="ok", duration_ms=10)
    base.update(overrides)
    return base


def _fail_result(error: str = "timeout", **overrides) -> dict:
    base = _result(ok=False, status="error", error=error, duration_ms=10)
    base.update(overrides)
    return base


def _decision(status: str = "allow") -> dict:
    return {
        "schema_version": BEADS_SCHEMA_VERSION,
        "decision_id": "d-1",
        "status": status,
        "summary": "test",
        "rationale": [],
        "required_constraints": [],
        "required_skills": [],
        "required_tests": [],
        "follow_up_goals": [],
        "target_subsystem": "general",
        "canonical_path": None,
        "overlap_classification": "not_targeted",
        "validation_status": "not_applicable",
        "required_remediation": [],
        "stop_reason": None,
    }


def _payload() -> dict:
    return {
        "schema_version": BEADS_SCHEMA_VERSION,
        "goal": "test goal",
        "goal_type": None,
        "runtime_mode": "queue",
        "project_root": "/tmp/test",
        "queue_summary": {},
        "active_context": {},
        "development_context": None,
        "prd_context": None,
        "conductor_track": None,
    }


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------

class TestRunWithRetry:
    def test_success_on_first_try(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        with patch.object(bridge, "run", return_value=_ok_result()) as mock_run:
            result = bridge.run_with_retry(_payload(), max_retries=2)
            assert result["ok"] is True
            assert mock_run.call_count == 1

    def test_retries_on_transient_then_succeeds(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        results = [
            _fail_result("timeout"),
            _ok_result(),
        ]
        with patch.object(bridge, "run", side_effect=results) as mock_run, \
             patch("core.beads_bridge.time.sleep"):
            result = bridge.run_with_retry(_payload(), max_retries=2, backoff_base=0.01)
            assert result["ok"] is True
            assert mock_run.call_count == 2

    def test_no_retry_on_non_transient(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        with patch.object(bridge, "run", return_value=_fail_result("invalid_json")) as mock_run:
            result = bridge.run_with_retry(_payload(), max_retries=3)
            assert result["ok"] is False
            assert mock_run.call_count == 1

    def test_all_retries_exhausted_queues_offline(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        with patch.object(bridge, "run", return_value=_fail_result("timeout")) as mock_run, \
             patch("core.beads_bridge.time.sleep"):
            result = bridge.run_with_retry(_payload(), max_retries=2, backoff_base=0.01)
            assert result["ok"] is False
            assert mock_run.call_count == 3  # 1 + 2 retries
        assert bridge.offline_queue_size() == 1

    def test_no_offline_queue_when_persist_disabled(self, tmp_path):
        bridge = _make_bridge(tmp_path, persist_artifacts=False)
        with patch.object(bridge, "run", return_value=_fail_result("timeout")), \
             patch("core.beads_bridge.time.sleep"):
            bridge.run_with_retry(_payload(), max_retries=1, backoff_base=0.01)
        assert bridge.offline_queue_size() == 0

    @pytest.mark.parametrize("error", list(_TRANSIENT_ERRORS))
    def test_all_transient_errors_trigger_retry(self, tmp_path, error):
        bridge = _make_bridge(tmp_path)
        with patch.object(bridge, "run", side_effect=[
            _fail_result(error),
            _ok_result(),
        ]) as mock_run, patch("core.beads_bridge.time.sleep"):
            result = bridge.run_with_retry(_payload(), max_retries=1, backoff_base=0.01)
            assert result["ok"] is True
            assert mock_run.call_count == 2

    def test_backoff_delays_increase(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        sleep_calls = []
        with patch.object(bridge, "run", return_value=_fail_result("timeout")), \
             patch("core.beads_bridge.time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            bridge.run_with_retry(_payload(), max_retries=3, backoff_base=1.0)
        assert len(sleep_calls) == 3
        assert sleep_calls[0] == 1.0   # 1.0 * 2^0
        assert sleep_calls[1] == 2.0   # 1.0 * 2^1
        assert sleep_calls[2] == 4.0   # 1.0 * 2^2

    def test_last_result_cached(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        ok = _ok_result()
        with patch.object(bridge, "run", return_value=ok):
            bridge.run_with_retry(_payload())
        assert bridge.last_result is ok


# ---------------------------------------------------------------------------
# Offline queue tests
# ---------------------------------------------------------------------------

class TestOfflineQueue:
    def test_enqueue_creates_file(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        (tmp_path / "memory").mkdir(exist_ok=True)
        bridge._enqueue_offline(_payload())
        assert bridge._offline_queue_path.exists()
        assert bridge.offline_queue_size() == 1

    def test_enqueue_appends(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        (tmp_path / "memory").mkdir(exist_ok=True)
        bridge._enqueue_offline(_payload())
        bridge._enqueue_offline(_payload())
        assert bridge.offline_queue_size() == 2

    def test_replay_succeeds_clears_queue(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        (tmp_path / "memory").mkdir(exist_ok=True)
        bridge._enqueue_offline(_payload())
        bridge._enqueue_offline(_payload())

        with patch.object(bridge, "run", return_value=_ok_result()):
            results = bridge.replay_offline_queue()
        assert len(results) == 2
        assert all(r["ok"] for r in results)
        assert bridge.offline_queue_size() == 0

    def test_replay_partial_failure_keeps_remaining(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        (tmp_path / "memory").mkdir(exist_ok=True)
        bridge._enqueue_offline(_payload())
        bridge._enqueue_offline(_payload())

        with patch.object(bridge, "run", side_effect=[
            _ok_result(),
            _fail_result("timeout"),
        ]):
            results = bridge.replay_offline_queue()
        assert len(results) == 2
        assert bridge.offline_queue_size() == 1

    def test_replay_empty_queue(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        results = bridge.replay_offline_queue()
        assert results == []

    def test_replay_skips_invalid_json_and_replays_valid_entries(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._offline_queue_path.parent.mkdir(exist_ok=True)
        bridge._offline_queue_path.write_text(
            "\n".join(
                [
                    "not-json",
                    json.dumps({"ts": time.time(), "payload": _payload()}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        with patch.object(bridge, "run", return_value=_ok_result()) as mock_run:
            results = bridge.replay_offline_queue()

        assert len(results) == 1
        assert mock_run.call_count == 1
        assert bridge.offline_queue_size() == 0
        assert bridge.offline_dead_letter_size() == 1

    def test_replay_dead_letters_entries_with_invalid_payload_shape(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._offline_queue_path.parent.mkdir(exist_ok=True)
        bridge._offline_queue_path.write_text(
            "\n".join(
                [
                    json.dumps({"ts": time.time(), "payload": "bad-payload"}),
                    json.dumps({"ts": time.time(), "payload": _payload()}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        with patch.object(bridge, "run", return_value=_ok_result()) as mock_run:
            results = bridge.replay_offline_queue()

        assert len(results) == 1
        assert mock_run.call_count == 1
        assert bridge.offline_queue_size() == 0
        assert bridge.offline_dead_letter_size() == 1

    def test_queue_size_zero_when_no_file(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        assert bridge.offline_queue_size() == 0


# ---------------------------------------------------------------------------
# Conflict detection tests
# ---------------------------------------------------------------------------

class TestConflictDetection:
    def test_no_conflict_when_no_last_result(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        assert bridge.has_conflict(_ok_result(decision=_decision("allow"))) is False

    def test_no_conflict_same_status(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge.last_result = _ok_result(decision=_decision("allow"))
        assert bridge.has_conflict(_ok_result(decision=_decision("allow"))) is False

    def test_conflict_different_status(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge.last_result = _ok_result(decision=_decision("allow"))
        assert bridge.has_conflict(_ok_result(decision=_decision("block"))) is True

    def test_no_conflict_when_old_has_no_decision(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge.last_result = _ok_result(decision=None)
        assert bridge.has_conflict(_ok_result(decision=_decision("block"))) is False

    def test_no_conflict_when_new_has_no_decision(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge.last_result = _ok_result(decision=_decision("allow"))
        assert bridge.has_conflict(_ok_result(decision=None)) is False

    def test_conflict_revise_vs_allow(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge.last_result = _ok_result(decision=_decision("revise"))
        assert bridge.has_conflict(_ok_result(decision=_decision("allow"))) is True


# ---------------------------------------------------------------------------
# Runtime metadata
# ---------------------------------------------------------------------------

class TestRuntimeMetadata:
    def test_includes_offline_queue_size(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        meta = bridge.to_runtime_metadata()
        assert "offline_queue_size" in meta
        assert meta["offline_queue_size"] == 0
        assert "offline_dead_letter_size" in meta
        assert meta["offline_dead_letter_size"] == 0
