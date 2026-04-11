"""Tests for agents/mutator.py — MutatorAgent."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agents.mutator import MutatorAgent


@pytest.fixture
def project_root(tmp_path):
    return tmp_path


@pytest.fixture
def agent(project_root):
    return MutatorAgent(project_root=project_root)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestMutatorInit:
    def test_project_root_stored(self, agent, project_root):
        assert agent.project_root == project_root

    def test_project_root_resolved(self, agent, project_root):
        assert agent.project_root_resolved == project_root.resolve()


# ---------------------------------------------------------------------------
# _validate_file_path
# ---------------------------------------------------------------------------

class TestValidateFilePath:
    def test_valid_relative_path(self, agent, project_root):
        (project_root / "subdir").mkdir()
        result = agent._validate_file_path("subdir/file.py")
        assert result == (project_root / "subdir" / "file.py").resolve()

    def test_absolute_path_raises(self, agent):
        with pytest.raises(ValueError, match="Absolute paths"):
            agent._validate_file_path("/etc/passwd")

    def test_path_traversal_raises(self, agent):
        with pytest.raises(ValueError, match="Path traversal"):
            agent._validate_file_path("../secret.txt")

    def test_nested_traversal_raises(self, agent):
        with pytest.raises(ValueError, match="Path traversal"):
            agent._validate_file_path("subdir/../../secret.txt")

    def test_valid_nested_path(self, agent, project_root):
        (project_root / "a" / "b").mkdir(parents=True)
        result = agent._validate_file_path("a/b/code.py")
        assert "code.py" in str(result)

    def test_simple_filename(self, agent, project_root):
        result = agent._validate_file_path("module.py")
        assert result.name == "module.py"
        assert str(project_root.resolve()) in str(result)


# ---------------------------------------------------------------------------
# apply_mutation — ADD_FILE block
# ---------------------------------------------------------------------------

class TestApplyMutationAddFile:
    def test_add_file_creates_file(self, agent, project_root):
        proposal = "ADD_FILE new_module.py\nprint('hello')\n"
        agent.apply_mutation(proposal)
        assert (project_root / "new_module.py").exists()

    def test_add_file_content_written(self, agent, project_root):
        proposal = "ADD_FILE output.py\nx = 42\n"
        agent.apply_mutation(proposal)
        content = (project_root / "output.py").read_text()
        assert "x = 42" in content

    def test_add_file_in_subdir(self, agent, project_root):
        (project_root / "pkg").mkdir()
        proposal = "ADD_FILE pkg/utils.py\ndef helper(): pass\n"
        agent.apply_mutation(proposal)
        assert (project_root / "pkg" / "utils.py").exists()

    def test_add_file_missing_path_logs_error(self, agent):
        proposal = "ADD_FILE\ncontent\n"
        # Should not raise; logs error internally
        agent.apply_mutation(proposal)

    def test_add_file_absolute_path_rejected(self, agent):
        proposal = "ADD_FILE /etc/evil.py\nbad content\n"
        agent.apply_mutation(proposal)
        assert not Path("/etc/evil.py").exists()


# ---------------------------------------------------------------------------
# apply_mutation — REPLACE_IN_FILE block
# ---------------------------------------------------------------------------

class TestApplyMutationReplaceInFile:
    def _make_replace_block(self, filepath, old, new):
        return (
            f"REPLACE_IN_FILE {filepath}\n"
            "---OLD_CONTENT_START---\n"
            f"{old}\n"
            "---OLD_CONTENT_END---\n"
            "---NEW_CONTENT_START---\n"
            f"{new}\n"
            "---NEW_CONTENT_END---\n"
        )

    def test_replace_content_in_existing_file(self, agent, project_root):
        target = project_root / "target.py"
        target.write_text("def foo():\n    return 1\n")
        block = self._make_replace_block("target.py", "def foo():\n    return 1", "def foo():\n    return 2")
        agent.apply_mutation(block)
        assert "return 2" in target.read_text()

    def test_replace_missing_markers_logs_error(self, agent, project_root):
        (project_root / "f.py").write_text("content")
        proposal = "REPLACE_IN_FILE f.py\nno markers here\n"
        # Should not raise
        agent.apply_mutation(proposal)

    def test_replace_with_traversal_path_rejected(self, agent, project_root):
        block = self._make_replace_block("../outside.py", "old", "new")
        agent.apply_mutation(block)
        assert not (project_root.parent / "outside.py").exists()


# ---------------------------------------------------------------------------
# apply_mutation — JSON mutation plan
# ---------------------------------------------------------------------------

class TestApplyMutationJson:
    def test_json_add_file(self, agent, project_root):
        plan = {
            "mutations": [
                {"type": "add_file", "file_path": "gen.py", "new_content": "# generated\n", "old_code": ""}
            ]
        }
        agent.apply_mutation(json.dumps(plan))
        assert (project_root / "gen.py").exists()

    def test_json_file_change(self, agent, project_root):
        target = project_root / "existing.py"
        target.write_text("x = 1\n")
        plan = {
            "mutations": [
                {"type": "file_change", "file_path": "existing.py", "new_content": "x = 2\n", "old_code": "x = 1\n"}
            ]
        }
        agent.apply_mutation(json.dumps(plan))
        assert "x = 2" in target.read_text()

    def test_json_missing_file_path_skipped(self, agent, project_root):
        plan = {"mutations": [{"type": "add_file", "new_content": "x"}]}
        agent.apply_mutation(json.dumps(plan))  # Should not raise

    def test_json_missing_new_content_skipped(self, agent, project_root):
        plan = {"mutations": [{"type": "add_file", "file_path": "f.py"}]}
        agent.apply_mutation(json.dumps(plan))  # Should not raise

    def test_json_traversal_rejected(self, agent, project_root):
        plan = {
            "mutations": [
                {"type": "add_file", "file_path": "../escape.py", "new_content": "evil", "old_code": ""}
            ]
        }
        agent.apply_mutation(json.dumps(plan))
        assert not (project_root.parent / "escape.py").exists()

    def test_multiple_json_mutations(self, agent, project_root):
        plan = {
            "mutations": [
                {"type": "add_file", "file_path": "a.py", "new_content": "# a\n", "old_code": ""},
                {"type": "add_file", "file_path": "b.py", "new_content": "# b\n", "old_code": ""},
            ]
        }
        agent.apply_mutation(json.dumps(plan))
        assert (project_root / "a.py").exists()
        assert (project_root / "b.py").exists()


# ---------------------------------------------------------------------------
# apply_mutation — empty / unrecognised input
# ---------------------------------------------------------------------------

class TestApplyMutationEdgeCases:
    def test_empty_proposal(self, agent):
        agent.apply_mutation("")  # Should not raise

    def test_whitespace_only_proposal(self, agent):
        agent.apply_mutation("   \n  ")  # Should not raise

    def test_unrecognised_block_skipped(self, agent):
        agent.apply_mutation("UNKNOWN_COMMAND some args\ncontent\n")  # Should not raise

    def test_multiple_blocks(self, agent, project_root):
        proposal = (
            "ADD_FILE first.py\n# first file\n\n"
            "ADD_FILE second.py\n# second file\n"
        )
        agent.apply_mutation(proposal)
        assert (project_root / "first.py").exists()
        assert (project_root / "second.py").exists()
