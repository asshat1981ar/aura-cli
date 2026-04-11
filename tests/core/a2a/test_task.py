"""Tests for core/a2a/task.py — A2AMessage, A2ATask, TaskState."""

import pytest

from core.a2a.task import A2AMessage, A2ATask, TaskState


# ---------------------------------------------------------------------------
# TaskState
# ---------------------------------------------------------------------------

class TestTaskState:
    def test_values(self):
        assert TaskState.SUBMITTED == "submitted"
        assert TaskState.WORKING == "working"
        assert TaskState.COMPLETED == "completed"
        assert TaskState.FAILED == "failed"
        assert TaskState.CANCELED == "canceled"
        assert TaskState.INPUT_REQUIRED == "input-required"

    def test_is_string_enum(self):
        assert isinstance(TaskState.SUBMITTED, str)


# ---------------------------------------------------------------------------
# A2AMessage
# ---------------------------------------------------------------------------

class TestA2AMessage:
    def test_role_and_content(self):
        msg = A2AMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_default_empty_parts(self):
        msg = A2AMessage(role="agent", content="response")
        assert msg.parts == []

    def test_timestamp_set(self):
        msg = A2AMessage(role="user", content="x")
        assert msg.timestamp > 0

    def test_custom_parts(self):
        msg = A2AMessage(role="agent", content="y", parts=[{"type": "text"}])
        assert len(msg.parts) == 1


# ---------------------------------------------------------------------------
# A2ATask — init
# ---------------------------------------------------------------------------

class TestA2ATaskInit:
    def test_id_auto_generated(self):
        task = A2ATask()
        assert task.id is not None
        assert len(task.id) == 36  # UUID4

    def test_default_state_submitted(self):
        task = A2ATask()
        assert task.state == TaskState.SUBMITTED

    def test_empty_messages_and_artifacts(self):
        task = A2ATask()
        assert task.messages == []
        assert task.artifacts == []

    def test_capability_stored(self):
        task = A2ATask(capability="code_generation")
        assert task.capability == "code_generation"


# ---------------------------------------------------------------------------
# A2ATask.transition
# ---------------------------------------------------------------------------

class TestA2ATaskTransition:
    def test_submitted_to_working(self):
        task = A2ATask()
        task.transition(TaskState.WORKING)
        assert task.state == TaskState.WORKING

    def test_submitted_to_canceled(self):
        task = A2ATask()
        task.transition(TaskState.CANCELED)
        assert task.state == TaskState.CANCELED

    def test_submitted_to_failed(self):
        task = A2ATask()
        task.transition(TaskState.FAILED)
        assert task.state == TaskState.FAILED

    def test_working_to_completed(self):
        task = A2ATask()
        task.transition(TaskState.WORKING)
        task.transition(TaskState.COMPLETED)
        assert task.state == TaskState.COMPLETED

    def test_working_to_input_required(self):
        task = A2ATask()
        task.transition(TaskState.WORKING)
        task.transition(TaskState.INPUT_REQUIRED)
        assert task.state == TaskState.INPUT_REQUIRED

    def test_input_required_to_working(self):
        task = A2ATask()
        task.transition(TaskState.WORKING)
        task.transition(TaskState.INPUT_REQUIRED)
        task.transition(TaskState.WORKING)
        assert task.state == TaskState.WORKING

    def test_invalid_transition_raises(self):
        task = A2ATask()
        with pytest.raises(ValueError, match="Invalid transition"):
            task.transition(TaskState.COMPLETED)  # submitted -> completed not allowed

    def test_terminal_state_no_transitions(self):
        task = A2ATask()
        task.transition(TaskState.WORKING)
        task.transition(TaskState.COMPLETED)
        with pytest.raises(ValueError):
            task.transition(TaskState.FAILED)

    def test_updated_at_changes_on_transition(self):
        task = A2ATask()
        before = task.updated_at
        task.transition(TaskState.WORKING)
        assert task.updated_at >= before


# ---------------------------------------------------------------------------
# A2ATask.add_message
# ---------------------------------------------------------------------------

class TestA2ATaskAddMessage:
    def test_message_appended(self):
        task = A2ATask()
        task.add_message("user", "hello")
        assert len(task.messages) == 1
        assert task.messages[0].role == "user"
        assert task.messages[0].content == "hello"

    def test_multiple_messages(self):
        task = A2ATask()
        task.add_message("user", "hi")
        task.add_message("agent", "response")
        assert len(task.messages) == 2

    def test_parts_passed_through(self):
        task = A2ATask()
        task.add_message("agent", "x", parts=[{"type": "code"}])
        assert task.messages[0].parts == [{"type": "code"}]

    def test_no_parts_defaults_empty(self):
        task = A2ATask()
        task.add_message("user", "msg")
        assert task.messages[0].parts == []


# ---------------------------------------------------------------------------
# A2ATask.add_artifact
# ---------------------------------------------------------------------------

class TestA2ATaskAddArtifact:
    def test_artifact_appended(self):
        task = A2ATask()
        task.add_artifact("result", {"key": "val"})
        assert len(task.artifacts) == 1
        assert task.artifacts[0]["name"] == "result"
        assert task.artifacts[0]["content"] == {"key": "val"}

    def test_default_mime_type(self):
        task = A2ATask()
        task.add_artifact("x", "content")
        assert task.artifacts[0]["mime_type"] == "application/json"

    def test_custom_mime_type(self):
        task = A2ATask()
        task.add_artifact("file", b"bytes", mime_type="text/plain")
        assert task.artifacts[0]["mime_type"] == "text/plain"


# ---------------------------------------------------------------------------
# A2ATask.to_dict
# ---------------------------------------------------------------------------

class TestA2ATaskToDict:
    def test_required_keys(self):
        task = A2ATask(capability="test_cap")
        d = task.to_dict()
        for key in ("id", "capability", "state", "messages", "artifacts",
                    "metadata", "created_at", "updated_at"):
            assert key in d

    def test_state_is_string(self):
        task = A2ATask()
        assert task.to_dict()["state"] == "submitted"

    def test_messages_serialized(self):
        task = A2ATask()
        task.add_message("user", "hello")
        d = task.to_dict()
        assert len(d["messages"]) == 1
        assert d["messages"][0]["role"] == "user"
        assert d["messages"][0]["content"] == "hello"

    def test_artifacts_included(self):
        task = A2ATask()
        task.add_artifact("output", {"result": "done"})
        d = task.to_dict()
        assert len(d["artifacts"]) == 1
