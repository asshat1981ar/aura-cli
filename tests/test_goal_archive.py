"""Tests for core/goal_archive.py — GoalArchive."""

import json
import pytest
from pathlib import Path

from core.goal_archive import GoalArchive


@pytest.fixture
def archive_path(tmp_path):
    return tmp_path / "goal_archive.json"


@pytest.fixture
def archive(archive_path):
    return GoalArchive(archive_path=str(archive_path))


class TestGoalArchiveInit:
    def test_starts_empty_no_file(self, archive):
        assert archive.completed == []

    def test_archive_path_stored_as_path(self, archive_path):
        a = GoalArchive(archive_path=str(archive_path))
        assert isinstance(a.archive_path, Path)

    def test_loads_existing_archive(self, archive_path):
        archive_path.write_text(json.dumps([["my goal", 0.9]]))
        a = GoalArchive(archive_path=str(archive_path))
        assert len(a.completed) == 1
        assert a.completed[0][0] == "my goal"

    def test_corrupted_json_starts_empty(self, archive_path):
        archive_path.write_text("{BROKEN JSON")
        a = GoalArchive(archive_path=str(archive_path))
        assert a.completed == []


class TestGoalArchiveRecord:
    def test_record_appends_entry(self, archive):
        archive.record("goal_1", 0.85)
        assert len(archive.completed) == 1
        goal, score = archive.completed[0]
        assert goal == "goal_1"
        assert score == 0.85

    def test_record_persists_to_disk(self, archive, archive_path):
        archive.record("goal_1", 1.0)
        data = json.loads(archive_path.read_text())
        assert data[0][0] == "goal_1"

    def test_record_multiple_entries(self, archive):
        archive.record("a", 0.5)
        archive.record("b", 0.9)
        assert len(archive.completed) == 2

    def test_record_preserves_insertion_order(self, archive):
        archive.record("first", 0.1)
        archive.record("second", 0.9)
        assert archive.completed[0][0] == "first"
        assert archive.completed[1][0] == "second"

    def test_record_with_dict_goal(self, archive):
        goal = {"title": "add auth", "priority": "high"}
        archive.record(goal, 0.7)
        assert archive.completed[0][0] == goal

    def test_record_score_stored_correctly(self, archive):
        archive.record("g", 0.42)
        assert archive.completed[0][1] == 0.42


class TestGoalArchivePersistence:
    def test_reload_after_record(self, archive, archive_path):
        archive.record("g1", 1.0)
        a2 = GoalArchive(archive_path=str(archive_path))
        assert len(a2.completed) == 1

    def test_parent_dir_created_if_needed(self, tmp_path):
        nested = tmp_path / "a" / "b" / "archive.json"
        a = GoalArchive(archive_path=str(nested))
        a.record("goal", 0.5)
        assert nested.exists()
