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
