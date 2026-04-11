"""Tests for aura/recording/recorder.py — RecordingSession, Recorder."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aura.recording.recorder import RecordingSession, Recorder
from aura.recording.models import Recording, RecordingStep, StepStatus


# ---------------------------------------------------------------------------
# RecordingSession
# ---------------------------------------------------------------------------

class TestRecordingSessionContext:
    def test_context_manager_sets_active(self):
        session = RecordingSession("test")
        assert session._active is False
        with session:
            assert session._active is True
        assert session._active is False

    def test_context_manager_returns_self(self):
        session = RecordingSession("test")
        with session as s:
            assert s is session

    def test_context_manager_not_suppress_exceptions(self):
        session = RecordingSession("test")
        with pytest.raises(ValueError):
            with session:
                raise ValueError("boom")

    def test_active_false_after_exception(self):
        session = RecordingSession("test")
        try:
            with session:
                raise RuntimeError("oops")
        except RuntimeError:
            pass
        assert session._active is False


class TestRecordingSessionRecordStep:
    def test_record_step_requires_active(self):
        session = RecordingSession("test")
        with pytest.raises(RuntimeError, match="not active"):
            session.record_step("my_cmd")

    def test_record_step_returns_step(self):
        session = RecordingSession("test")
        with session:
            step = session.record_step("run_cmd", "arg1")
        assert isinstance(step, RecordingStep)
        assert step.command == "run_cmd"

    def test_record_step_args_stored(self):
        session = RecordingSession("test")
        with session:
            step = session.record_step("cmd", "a", "b", "c")
        assert step.args == ["a", "b", "c"]

    def test_record_step_kwargs_stored(self):
        session = RecordingSession("test")
        with session:
            step = session.record_step("cmd", key="val")
        assert step.kwargs == {"key": "val"}

    def test_record_step_condition_stored(self):
        session = RecordingSession("test")
        with session:
            step = session.record_step("cmd", condition="x == y")
        assert step.condition == "x == y"

    def test_record_step_retry_defaults(self):
        session = RecordingSession("test")
        with session:
            step = session.record_step("cmd")
        assert step.retry_count == 3
        assert step.retry_delay == 1.0
        assert step.timeout == 60

    def test_record_step_custom_retry(self):
        session = RecordingSession("test")
        with session:
            step = session.record_step("cmd", retry_count=5, retry_delay=0.5, timeout=30)
        assert step.retry_count == 5
        assert step.retry_delay == 0.5
        assert step.timeout == 30

    def test_record_step_adds_to_recording(self):
        session = RecordingSession("test")
        with session:
            session.record_step("cmd_a")
            session.record_step("cmd_b")
        assert session.recording.step_count == 2


class TestRecordingSessionVariables:
    def test_set_variable_stores_in_recording(self):
        session = RecordingSession("test")
        session.set_variable("env", "prod")
        assert session.recording.variables["env"] == "prod"

    def test_set_multiple_variables(self):
        session = RecordingSession("test")
        session.set_variable("a", "1")
        session.set_variable("b", "2")
        assert session.recording.variables == {"a": "1", "b": "2"}


class TestRecordingSessionSave:
    async def test_save_returns_path_string(self, tmp_path):
        from aura.recording.storage import RecordingStorage
        storage = RecordingStorage(directory=tmp_path)
        session = RecordingSession("save_test", storage=storage)
        with session:
            session.record_step("cmd")
        path = await session.save()
        assert isinstance(path, str)
        assert "save_test" in path


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class TestRecorderStartRecording:
    def test_start_recording_returns_session(self):
        recorder = Recorder()
        session = recorder.start_recording("my_session")
        assert isinstance(session, RecordingSession)

    def test_start_recording_uses_given_name(self):
        recorder = Recorder()
        session = recorder.start_recording("hello")
        assert session.recording.name == "hello"

    def test_start_recording_shares_storage(self, tmp_path):
        from aura.recording.storage import RecordingStorage
        storage = RecordingStorage(directory=tmp_path)
        recorder = Recorder(storage=storage)
        session = recorder.start_recording("test")
        assert session.storage is storage


class TestRecorderCRUD:
    async def test_list_recordings_empty(self, tmp_path):
        from aura.recording.storage import RecordingStorage
        storage = RecordingStorage(directory=tmp_path)
        recorder = Recorder(storage=storage)
        recordings = await recorder.list_recordings()
        assert recordings == []

    async def test_exists_false_before_save(self, tmp_path):
        from aura.recording.storage import RecordingStorage
        storage = RecordingStorage(directory=tmp_path)
        recorder = Recorder(storage=storage)
        assert await recorder.exists("nosuchrecording") is False

    async def test_save_and_exists(self, tmp_path):
        from aura.recording.storage import RecordingStorage
        storage = RecordingStorage(directory=tmp_path)
        recorder = Recorder(storage=storage)
        session = recorder.start_recording("myrecording")
        with session:
            session.record_step("noop")
        await session.save()
        assert await recorder.exists("myrecording") is True

    async def test_load_returns_recording(self, tmp_path):
        from aura.recording.storage import RecordingStorage
        storage = RecordingStorage(directory=tmp_path)
        recorder = Recorder(storage=storage)
        session = recorder.start_recording("loadme")
        with session:
            session.record_step("cmd")
        await session.save()
        loaded = await recorder.load("loadme")
        assert loaded is not None
        assert loaded.name == "loadme"

    async def test_load_nonexistent_returns_none(self, tmp_path):
        from aura.recording.storage import RecordingStorage
        storage = RecordingStorage(directory=tmp_path)
        recorder = Recorder(storage=storage)
        result = await recorder.load("ghost")
        assert result is None

    async def test_delete_existing_returns_true(self, tmp_path):
        from aura.recording.storage import RecordingStorage
        storage = RecordingStorage(directory=tmp_path)
        recorder = Recorder(storage=storage)
        session = recorder.start_recording("todelete")
        with session:
            session.record_step("x")
        await session.save()
        assert await recorder.delete("todelete") is True
        assert await recorder.exists("todelete") is False

    async def test_delete_nonexistent_returns_false(self, tmp_path):
        from aura.recording.storage import RecordingStorage
        storage = RecordingStorage(directory=tmp_path)
        recorder = Recorder(storage=storage)
        assert await recorder.delete("phantom") is False

    async def test_list_after_save(self, tmp_path):
        from aura.recording.storage import RecordingStorage
        storage = RecordingStorage(directory=tmp_path)
        recorder = Recorder(storage=storage)
        session = recorder.start_recording("listed")
        with session:
            session.record_step("cmd")
        await session.save()
        recordings = await recorder.list_recordings()
        assert len(recordings) == 1
        assert recordings[0].name == "listed"
