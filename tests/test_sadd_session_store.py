"""Unit tests for core.sadd.session_store.SessionStore."""

from __future__ import annotations

import time
import unittest

import pytest

from core.sadd.session_store import SessionStore
from core.sadd.types import (
    DesignSpec,
    SessionConfig,
    SessionReport,
    WorkstreamOutcome,
    WorkstreamSpec,
)


def _make_spec(title: str = "Test Spec") -> DesignSpec:
    ws = WorkstreamSpec(id="ws_a", title="Task A", goal_text="Do task A")
    return DesignSpec(title=title, summary="Test", workstreams=[ws])


def _make_config(**kwargs) -> SessionConfig:
    return SessionConfig(**kwargs)


def _make_report(session_id: str = "sid") -> SessionReport:
    return SessionReport(
        session_id=session_id,
        design_title="Test Spec",
        total_workstreams=1,
        completed=1,
        failed=0,
        skipped=0,
        outcomes=[WorkstreamOutcome(id="ws_a", title="Task A", status="completed", cycles_used=2)],
        elapsed_s=1.5,
    )


class TestSessionStoreCreateAndGet:
    """test_create_and_get_session — create a session, get it back, verify fields."""

    def test_create_and_get_session(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        spec = _make_spec("My Session")
        config = _make_config(dry_run=True)

        sid = store.create_session(spec, config, session_id="sess-1")

        assert sid == "sess-1"
        session = store.get_session("sess-1")
        assert session is not None
        assert session["id"] == "sess-1"
        assert session["title"] == "My Session"
        assert session["status"] == "pending"
        assert session["created_at"] > 0
        assert session["updated_at"] > 0

    def test_create_session_auto_id(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        sid = store.create_session(_make_spec(), _make_config())
        assert sid is not None
        assert len(sid) > 0
        session = store.get_session(sid)
        assert session is not None


class TestSessionStoreUpdateStatus:
    """test_update_status — create, update status to 'running', verify."""

    def test_update_status(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        sid = store.create_session(_make_spec(), _make_config(), session_id="s1")

        store.update_status("s1", "running")

        session = store.get_session("s1")
        assert session is not None
        assert session["status"] == "running"


class TestSessionStoreSaveReport:
    """test_save_and_load_report — create, save report, verify report_json is stored."""

    def test_save_and_load_report(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        sid = store.create_session(_make_spec(), _make_config(), session_id="s1")

        report = _make_report("s1")
        store.save_report("s1", report)

        session = store.get_session("s1")
        assert session is not None
        assert session["report_json"] is not None
        assert session["status"] == "completed"
        import json

        report_data = json.loads(session["report_json"])
        assert report_data["design_title"] == "Test Spec"
        assert report_data["completed"] == 1


class TestSessionStoreListSessions:
    """test_list_sessions — create 3 sessions, list returns them in reverse chronological order."""

    def test_list_sessions(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        for i in range(3):
            store.create_session(_make_spec(f"Session {i}"), _make_config(), session_id=f"s{i}")
            # Small sleep to ensure ordering by created_at
            time.sleep(0.01)

        sessions = store.list_sessions()
        assert len(sessions) == 3
        # Most recent first
        assert sessions[0]["id"] == "s2"
        assert sessions[1]["id"] == "s1"
        assert sessions[2]["id"] == "s0"


class TestSessionStoreCheckpoints:
    """Checkpoint save, load, and list tests."""

    def test_save_and_load_checkpoint(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        sid = store.create_session(_make_spec(), _make_config(), session_id="s1")

        cp1 = store.save_checkpoint("s1", {"phase": 1}, {"ws_a": "partial"})
        cp2 = store.save_checkpoint("s1", {"phase": 2}, {"ws_a": "done"})

        assert cp1 > 0
        assert cp2 > cp1

        latest = store.load_latest_checkpoint("s1")
        assert latest is not None
        graph_state, results = latest
        assert graph_state == {"phase": 2}
        assert results == {"ws_a": "done"}

    def test_list_checkpoints(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        store.create_session(_make_spec(), _make_config(), session_id="s1")

        store.save_checkpoint("s1", {"a": 1}, {"r": 1})
        store.save_checkpoint("s1", {"a": 2}, {"r": 2})
        store.save_checkpoint("s1", {"a": 3}, {"r": 3})

        cps = store.list_checkpoints("s1")
        assert len(cps) == 3
        # Ordered by id ascending
        assert cps[0]["id"] < cps[1]["id"] < cps[2]["id"]

    def test_no_checkpoint_returns_none(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        store.create_session(_make_spec(), _make_config(), session_id="s1")

        result = store.load_latest_checkpoint("s1")
        assert result is None


class TestSessionStoreEvents:
    """test_log_and_get_events — log multiple events, get all, get filtered by type."""

    def test_log_and_get_events(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        store.create_session(_make_spec(), _make_config(), session_id="s1")

        store.log_event("s1", "ws_a", "started", {"detail": "begin"})
        store.log_event("s1", "ws_a", "completed", {"detail": "end"})
        store.log_event("s1", None, "checkpoint", {"cp": 1})

        # Note: create_session also logs a session_created event, so total is 4
        all_events = store.get_events("s1")
        assert len(all_events) == 4

        started_events = store.get_events("s1", event_type="started")
        assert len(started_events) == 1
        assert started_events[0]["event_type"] == "started"

        checkpoint_events = store.get_events("s1", event_type="checkpoint")
        assert len(checkpoint_events) == 1


class TestSessionStoreArtifacts:
    """test_record_and_get_artifacts — record artifacts, get all, get filtered by ws_id."""

    def test_record_and_get_artifacts(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        store.create_session(_make_spec(), _make_config(), session_id="s1")

        store.record_artifact("s1", "ws_a", "/path/to/file1.py")
        store.record_artifact("s1", "ws_a", "/path/to/file2.py")
        store.record_artifact("s1", "ws_b", "/path/to/file3.py")

        all_artifacts = store.get_artifacts("s1")
        assert len(all_artifacts) == 3

        ws_a_artifacts = store.get_artifacts("s1", ws_id="ws_a")
        assert len(ws_a_artifacts) == 2
        assert all(a["ws_id"] == "ws_a" for a in ws_a_artifacts)

        ws_b_artifacts = store.get_artifacts("s1", ws_id="ws_b")
        assert len(ws_b_artifacts) == 1
        assert ws_b_artifacts[0]["file_path"] == "/path/to/file3.py"


class TestSessionStoreResumable:
    """test_resumable_sessions — create running and completed sessions, only running shows up."""

    def test_resumable_sessions(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")

        store.create_session(_make_spec("Running"), _make_config(), session_id="s_run")
        store.update_status("s_run", "running")

        store.create_session(_make_spec("Done"), _make_config(), session_id="s_done")
        store.update_status("s_done", "completed")

        store.create_session(_make_spec("Pending"), _make_config(), session_id="s_pending")

        resumable = store.resumable_sessions()
        resumable_ids = {s["id"] for s in resumable}
        assert "s_run" in resumable_ids
        assert "s_pending" in resumable_ids
        assert "s_done" not in resumable_ids


class TestSessionStoreLoadForResume:
    """test_load_session_for_resume — create session with checkpoint, load for resume."""

    def test_load_session_for_resume(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        spec = _make_spec("Resume Me")
        config = _make_config(max_parallel=2)
        store.create_session(spec, config, session_id="s1")
        store.save_checkpoint("s1", {"nodes": ["ws_a"]}, {"ws_a": "partial"})

        result = store.load_session_for_resume("s1")
        assert result is not None
        loaded_spec, loaded_config, graph_state, results = result

        assert loaded_spec.title == "Resume Me"
        assert loaded_config.max_parallel == 2
        assert graph_state == {"nodes": ["ws_a"]}
        assert results == {"ws_a": "partial"}

    def test_load_session_for_resume_no_checkpoint(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")
        store.create_session(_make_spec(), _make_config(), session_id="s1")

        result = store.load_session_for_resume("s1")
        assert result is not None
        _, _, graph_state, results = result
        assert graph_state == {}
        assert results == {}


class TestSessionStoreNonexistent:
    """test_nonexistent_session — get_session returns None, load_session_for_resume returns None."""

    def test_nonexistent_session(self, tmp_path):
        store = SessionStore(db_path=tmp_path / "test.db")

        assert store.get_session("does-not-exist") is None
        assert store.load_session_for_resume("does-not-exist") is None
