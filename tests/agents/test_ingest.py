"""Tests for agents/ingest.py — IngestAgent."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_brain():
    brain = MagicMock()
    brain.recall_recent.return_value = ["memory_entry_1", "memory_entry_2"]
    return brain


@pytest.fixture
def agent(mock_brain):
    from agents.ingest import IngestAgent

    return IngestAgent(brain=mock_brain)


@pytest.fixture
def agent_with_cm(mock_brain):
    from agents.ingest import IngestAgent

    cm = MagicMock()
    cm.get_context_bundle.return_value = {
        "memory": ["m1", "m2"],
        "files": ["file_a.py", "file_b.py"],
        "related_insights": ["hint1"],
    }
    cm.format_as_prompt.return_value = "formatted_prompt"
    return IngestAgent(brain=mock_brain, context_manager=cm), cm


class TestIngestAgentInit:
    def test_name(self, agent):
        assert agent.name == "ingest"

    def test_brain_stored(self, agent, mock_brain):
        assert agent.brain is mock_brain

    def test_no_context_manager_by_default(self, agent):
        assert agent.cm is None


class TestSnapshotMethod:
    def test_snapshot_returns_sorted_file_list(self, agent, tmp_path):
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "a.py").write_text("a")
        files = agent._snapshot(tmp_path)
        assert files == sorted(files)

    def test_snapshot_excludes_hidden_dirs(self, agent, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("x")
        (tmp_path / "main.py").write_text("x")
        files = agent._snapshot(tmp_path)
        assert not any(".git" in f for f in files)
        assert any("main.py" in f for f in files)

    def test_snapshot_excludes_pycache(self, agent, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.pyc").write_text("x")
        (tmp_path / "real.py").write_text("x")
        files = agent._snapshot(tmp_path)
        assert not any("__pycache__" in f for f in files)

    def test_snapshot_handles_missing_dir(self, agent, tmp_path):
        files = agent._snapshot(tmp_path / "nonexistent")
        assert files == []

    def test_snapshot_nested_files(self, agent, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("x")
        files = agent._snapshot(tmp_path)
        assert any("nested.py" in f for f in files)


class TestRunWithoutContextManager:
    def test_returns_dict(self, agent, tmp_path):
        result = agent.run({"goal": "do something", "project_root": str(tmp_path)})
        assert isinstance(result, dict)

    def test_goal_propagated(self, agent, tmp_path):
        result = agent.run({"goal": "my goal", "project_root": str(tmp_path)})
        assert result["goal"] == "my goal"

    def test_memory_summary_present(self, agent, tmp_path):
        result = agent.run({"goal": "x", "project_root": str(tmp_path)})
        assert "memory_summary" in result

    def test_snapshot_in_output(self, agent, tmp_path):
        (tmp_path / "code.py").write_text("x")
        result = agent.run({"goal": "x", "project_root": str(tmp_path)})
        assert "snapshot" in result

    def test_constraints_propagated(self, agent, tmp_path):
        result = agent.run({"goal": "x", "project_root": str(tmp_path), "constraints": {"k": "v"}})
        assert result["constraints"] == {"k": "v"}

    def test_hints_summarized(self, agent, tmp_path):
        hints = [{"cycle_id": "c1", "status": "ok", "stop_reason": None, "summary": "good run"}]
        result = agent.run({"goal": "x", "project_root": str(tmp_path), "hints": hints})
        assert "cycle=c1" in result["hints_summary"]

    def test_empty_hints(self, agent, tmp_path):
        result = agent.run({"goal": "x", "project_root": str(tmp_path), "hints": []})
        assert result["hints_summary"] == ""

    def test_non_dict_hints_skipped(self, agent, tmp_path):
        result = agent.run({"goal": "x", "project_root": str(tmp_path), "hints": ["not a dict"]})
        assert result["hints_summary"] == ""

    def test_brain_recall_called(self, agent, mock_brain, tmp_path):
        agent.run({"goal": "x", "project_root": str(tmp_path)})
        mock_brain.recall_recent.assert_called_once_with(limit=50)

    def test_default_project_root(self, agent):
        result = agent.run({"goal": "x"})
        assert "goal" in result


class TestRunWithContextManager:
    def test_uses_context_manager(self, agent_with_cm):
        agent, cm = agent_with_cm
        result = agent.run({"goal": "build feature", "project_root": "."})
        cm.get_context_bundle.assert_called_once()
        assert "bundle" in result
        assert result["prompt_segment"] == "formatted_prompt"

    def test_memory_summary_from_bundle(self, agent_with_cm):
        agent, cm = agent_with_cm
        result = agent.run({"goal": "x", "project_root": "."})
        assert "m1" in result["memory_summary"]

    def test_snapshot_from_bundle_files(self, agent_with_cm):
        agent, cm = agent_with_cm
        result = agent.run({"goal": "x", "project_root": "."})
        assert "file_a.py" in result["snapshot"]

    def test_hints_summary_from_bundle(self, agent_with_cm):
        agent, cm = agent_with_cm
        result = agent.run({"goal": "x", "project_root": "."})
        assert "hint1" in result["hints_summary"]
