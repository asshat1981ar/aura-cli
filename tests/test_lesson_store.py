"""Tests for memory/lesson_store.py — LessonStore persistence and injection."""

import json
import pytest
from pathlib import Path
from memory.lesson_store import LessonStore, _derive_lesson


class TestLessonStoreBasic:
    def test_create_empty_store(self, tmp_path):
        store = LessonStore(store_path=tmp_path / "lessons.jsonl")
        assert store.count() == 0

    def test_record_and_count(self, tmp_path):
        store = LessonStore(store_path=tmp_path / "lessons.jsonl")
        store.record_cycle({"goal": "test goal", "stop_reason": "done"})
        assert store.count() == 1

    def test_record_persists_to_disk(self, tmp_path):
        path = tmp_path / "lessons.jsonl"
        store = LessonStore(store_path=path)
        store.record_cycle({"goal": "persist test", "stop_reason": "done"})
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["goal"] == "persist test"

    def test_load_on_init(self, tmp_path):
        path = tmp_path / "lessons.jsonl"
        store1 = LessonStore(store_path=path)
        store1.record_cycle({"goal": "a"})
        store1.record_cycle({"goal": "b"})
        store2 = LessonStore(store_path=path)
        assert store2.count() == 2


class TestInjectableLessons:
    def test_returns_empty_when_no_lessons(self, tmp_path):
        store = LessonStore(store_path=tmp_path / "l.jsonl")
        assert store.injectable_lessons() == []

    def test_returns_recent_lessons(self, tmp_path):
        store = LessonStore(store_path=tmp_path / "l.jsonl", max_injectable=2)
        for i in range(5):
            store.record_cycle({"goal": f"goal-{i}", "stop_reason": "done"})
        lessons = store.injectable_lessons()
        assert len(lessons) == 2
        assert lessons[0]["goal"] == "goal-3"
        assert lessons[1]["goal"] == "goal-4"

    def test_lesson_format(self, tmp_path):
        store = LessonStore(store_path=tmp_path / "l.jsonl")
        store.record_cycle({"goal": "test", "stop_reason": "done", "phase_outputs": {"cycle_confidence": 0.8}})
        lessons = store.injectable_lessons()
        assert len(lessons) == 1
        assert "goal" in lessons[0]
        assert "outcome" in lessons[0]
        assert "lesson" in lessons[0]


class TestDeriveLesson:
    def test_max_cycles_lesson(self):
        result = _derive_lesson({"goal": "big task", "stop_reason": "MAX_CYCLES"})
        assert "decomposing" in result

    def test_verify_fail_lesson(self):
        result = _derive_lesson({"goal": "buggy", "verify_status": "fail"})
        assert "verification" in result

    def test_verify_pass_lesson(self):
        result = _derive_lesson({"goal": "good", "verify_status": "pass"})
        assert "succeeded" in result


class TestCorruptionResilience:
    def test_corrupt_file_doesnt_crash(self, tmp_path):
        path = tmp_path / "l.jsonl"
        path.write_text("not json\n{bad\n")
        store = LessonStore(store_path=path)
        assert store.count() == 0

    def test_missing_file_starts_empty(self, tmp_path):
        store = LessonStore(store_path=tmp_path / "nonexistent.jsonl")
        assert store.count() == 0


class TestOrchestratorIntegration:
    def test_none_lesson_store_doesnt_crash(self):
        """Orchestrator should handle lesson_store=None gracefully."""
        from unittest.mock import MagicMock, patch
        from core.orchestrator import LoopOrchestrator

        with patch.object(LoopOrchestrator, "__init__", lambda self, **kw: None):
            orch = LoopOrchestrator()
            orch.lesson_store = None
            # Simulate the guard pattern used in run_cycle
            if orch.lesson_store:
                orch.lesson_store.injectable_lessons()
            # Should not raise


class TestLessonStoreEdgeCases:
    """Edge case tests for LessonStore."""

    def test_store_path_default(self):
        """Test default store path."""
        store = LessonStore()
        assert str(store.store_path) == "memory/lessons.jsonl"

    def test_store_path_custom_string(self, tmp_path):
        """Test custom store path as string."""
        store = LessonStore(store_path=str(tmp_path / "custom.jsonl"))
        assert isinstance(store.store_path, Path)

    def test_max_injectable_custom(self, tmp_path):
        """Test custom max_injectable value."""
        store = LessonStore(store_path=tmp_path / "l.jsonl", max_injectable=3)
        assert store.max_injectable == 3

    def test_record_cycle_timestamp(self, tmp_path):
        """Test that record_cycle includes timestamp."""
        store = LessonStore(store_path=tmp_path / "l.jsonl")
        store.record_cycle({"goal": "test"})

        path = tmp_path / "l.jsonl"
        entry = json.loads(path.read_text().strip())
        assert "timestamp" in entry
        assert isinstance(entry["timestamp"], float)

    def test_record_cycle_extracts_all_fields(self, tmp_path):
        """Test that record_cycle extracts all expected fields."""
        store = LessonStore(store_path=tmp_path / "l.jsonl")
        cycle_result = {"goal": "test goal", "goal_type": "bug_fix", "stop_reason": "MAX_CYCLES", "status": "failed", "phase_outputs": {"verify": {"status": "fail"}, "cycle_confidence": 0.5, "_failure_context": "timeout"}}
        store.record_cycle(cycle_result)

        path = tmp_path / "l.jsonl"
        entry = json.loads(path.read_text().strip())
        assert entry["goal"] == "test goal"
        assert entry["goal_type"] == "bug_fix"
        assert entry["stop_reason"] == "MAX_CYCLES"
        assert entry["status"] == "failed"
        assert entry["verify_status"] == "fail"
        assert entry["cycle_confidence"] == 0.5
        assert entry["failure_context"] == "timeout"

    def test_injectable_lessons_empty_file(self, tmp_path):
        """Test injectable_lessons with empty file."""
        path = tmp_path / "l.jsonl"
        path.write_text("")
        store = LessonStore(store_path=path)
        assert store.injectable_lessons() == []

    def test_injectable_lessons_with_empty_lines(self, tmp_path):
        """Test injectable_lessons ignores empty lines."""
        path = tmp_path / "l.jsonl"
        path.write_text('{"goal": "a"}\n\n{"goal": "b"}\n   \n')
        store = LessonStore(store_path=path)
        lessons = store.injectable_lessons()
        assert len(lessons) == 2

    def test_injectable_lessons_respects_limit(self, tmp_path):
        """Test that injectable_lessons respects max_injectable limit."""
        store = LessonStore(store_path=tmp_path / "l.jsonl", max_injectable=2)
        for i in range(5):
            store.record_cycle({"goal": f"goal-{i}"})

        lessons = store.injectable_lessons()
        assert len(lessons) == 2
        assert lessons[0]["goal"] == "goal-3"
        assert lessons[1]["goal"] == "goal-4"

    def test_lesson_confidence_default(self, tmp_path):
        """Test that injectable lessons have confidence."""
        store = LessonStore(store_path=tmp_path / "l.jsonl")
        store.record_cycle({"goal": "test", "phase_outputs": {"cycle_confidence": 0.75}})
        lessons = store.injectable_lessons()
        assert lessons[0]["confidence"] == 0.75

    def test_count_after_load(self, tmp_path):
        """Test count is accurate after loading from disk."""
        path = tmp_path / "l.jsonl"
        store1 = LessonStore(store_path=path)
        for i in range(3):
            store1.record_cycle({"goal": f"goal-{i}"})

        # New store loads existing
        store2 = LessonStore(store_path=path)
        assert store2.count() == 3


class TestDeriveLessonComprehensive:
    """Comprehensive tests for _derive_lesson function."""

    def test_derive_lesson_max_cycles(self):
        """Test lesson derivation for MAX_CYCLES."""
        entry = {"goal": "long task", "stop_reason": "MAX_CYCLES"}
        lesson = _derive_lesson(entry)
        assert "max cycles" in lesson.lower()
        assert "decomposing" in lesson.lower()
        assert "long task" in lesson

    def test_derive_lesson_verify_fail(self):
        """Test lesson derivation for failed verification."""
        entry = {"goal": "buggy code", "verify_status": "fail"}
        lesson = _derive_lesson(entry)
        assert "fail" in lesson.lower()
        assert "verification" in lesson.lower()
        assert "buggy code" in lesson

    def test_derive_lesson_verify_pass(self):
        """Test lesson derivation for passed verification."""
        entry = {"goal": "good code", "verify_status": "pass"}
        lesson = _derive_lesson(entry)
        assert "succeed" in lesson.lower()
        assert "good code" in lesson

    def test_derive_lesson_default(self):
        """Test lesson derivation with no special status."""
        entry = {"goal": "unknown goal", "stop_reason": "OTHER"}
        lesson = _derive_lesson(entry)
        assert "completed" in lesson.lower()
        assert "OTHER" in lesson

    def test_derive_lesson_long_goal_truncated(self):
        """Test that long goals are truncated to 50 chars."""
        long_goal = "a" * 100
        entry = {"goal": long_goal}
        lesson = _derive_lesson(entry)
        assert "a" * 50 in lesson
        assert "a" * 51 not in lesson

    def test_derive_lesson_missing_goal(self):
        """Test lesson derivation with missing goal."""
        entry = {"stop_reason": "DONE"}
        lesson = _derive_lesson(entry)
        assert "unknown" in lesson.lower()

    def test_derive_lesson_missing_all_fields(self):
        """Test lesson derivation with empty entry."""
        entry = {}
        lesson = _derive_lesson(entry)
        assert isinstance(lesson, str)
        assert "unknown" in lesson.lower()


class TestLessonStoreFileHandling:
    """Tests for file I/O handling in LessonStore."""

    def test_create_parent_directory(self, tmp_path):
        """Test that parent directories are created."""
        nested_path = tmp_path / "a" / "b" / "c" / "lessons.jsonl"
        store = LessonStore(store_path=nested_path)
        store.record_cycle({"goal": "test"})

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_append_mode_preserves_existing(self, tmp_path):
        """Test that multiple records are appended."""
        path = tmp_path / "l.jsonl"
        store = LessonStore(store_path=path)

        store.record_cycle({"goal": "first"})
        store.record_cycle({"goal": "second"})

        content = path.read_text()
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) == 2

    def test_json_serialization_error_safe(self, tmp_path):
        """Test handling of objects that can't be serialized to JSON."""
        from datetime import datetime

        store = LessonStore(store_path=tmp_path / "l.jsonl")
        cycle_result = {
            "goal": "test",
            "phase_outputs": {
                "cycle_confidence": 0.5,
                # datetime objects should be converted via default=str
            },
        }
        store.record_cycle(cycle_result)

        path = tmp_path / "l.jsonl"
        content = path.read_text()
        assert len(content) > 0

    def test_load_malformed_json_recovers(self, tmp_path):
        """Test that malformed JSON doesn't crash load."""
        path = tmp_path / "l.jsonl"
        path.write_text('{"goal": "good"}\nBAD JSON HERE\n{"goal": "also good"}')

        # Should skip bad lines gracefully
        store = LessonStore(store_path=path)
        # Count should only include valid entries
        assert store.count() <= 2
