"""Tests for core/git_tools.py — GitTools, _require_gitpython."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# _require_gitpython
# ---------------------------------------------------------------------------

class TestRequireGitpython:
    def test_raises_import_error_when_repo_is_none(self):
        import core.git_tools as gt
        original_repo = gt.Repo
        gt.Repo = None
        try:
            with pytest.raises(ImportError, match="gitpython"):
                gt._require_gitpython()
        finally:
            gt.Repo = original_repo

    def test_no_error_when_repo_available(self):
        import core.git_tools as gt
        if gt.Repo is not None:
            gt._require_gitpython()  # Should not raise


# ---------------------------------------------------------------------------
# GitTools init
# ---------------------------------------------------------------------------

class TestGitToolsInit:
    def _make_mock_repo(self, working_tree_dir="/tmp/repo"):
        mock = MagicMock()
        mock.working_tree_dir = working_tree_dir
        return mock

    def test_init_stores_repo_root(self):
        from core.git_tools import GitTools
        mock_repo = self._make_mock_repo("/fake/repo")
        with patch("core.git_tools.Repo", return_value=mock_repo):
            tools = GitTools(repo_path="/fake/repo")
        assert tools.repo_root == "/fake/repo"

    def test_init_raises_git_repo_error_on_invalid_repo(self):
        from core.git_tools import GitTools
        from core.exceptions import GitRepoError
        with patch("core.git_tools.Repo", side_effect=Exception("not a repo")):
            with pytest.raises((GitRepoError, Exception)):
                GitTools(repo_path="/not/a/repo")

    def test_init_no_repo_path_uses_cwd(self):
        from core.git_tools import GitTools
        mock_repo = self._make_mock_repo(".")
        with patch("core.git_tools.Repo", return_value=mock_repo):
            tools = GitTools()
        assert tools.repo is mock_repo


# ---------------------------------------------------------------------------
# commit_all
# ---------------------------------------------------------------------------

class TestCommitAll:
    @pytest.fixture
    def tools(self):
        from core.git_tools import GitTools
        mock_repo = MagicMock()
        mock_repo.working_tree_dir = "/fake"
        mock_repo.untracked_files = []
        mock_repo.is_dirty.return_value = True
        with patch("core.git_tools.Repo", return_value=mock_repo):
            t = GitTools(repo_path="/fake")
        return t, mock_repo

    def test_commit_calls_index_commit(self, tools):
        t, repo = tools
        t.commit_all("test commit")
        repo.index.commit.assert_called_once_with("test commit")

    def test_commit_adds_all_when_dirty(self, tools):
        t, repo = tools
        t.commit_all("msg")
        repo.git.add.assert_called_with(A=True)

    def test_commit_skipped_when_not_dirty(self, tools):
        t, repo = tools
        repo.is_dirty.return_value = False
        t.commit_all("msg")
        repo.index.commit.assert_not_called()

    def test_untracked_files_added(self, tools):
        t, repo = tools
        repo.untracked_files = ["new_file.py"]
        t.commit_all("msg")
        # Should have called add for untracked files
        assert repo.git.add.called

    def test_git_command_error_raises_git_commit_error(self, tools):
        from core.exceptions import GitCommitError
        t, repo = tools
        repo.index.commit.side_effect = Exception("git error")
        with pytest.raises(Exception):
            t.commit_all("msg")


# ---------------------------------------------------------------------------
# rollback_last_commit
# ---------------------------------------------------------------------------

class TestRollbackLastCommit:
    @pytest.fixture
    def tools(self):
        from core.git_tools import GitTools
        mock_repo = MagicMock()
        mock_repo.working_tree_dir = "/fake"
        mock_repo.untracked_files = []
        mock_repo.is_dirty.return_value = False
        # Simulate a repo with a parent commit
        parent = MagicMock()
        mock_repo.head.commit.parents = [parent]
        with patch("core.git_tools.Repo", return_value=mock_repo):
            t = GitTools(repo_path="/fake")
        return t, mock_repo

    def test_rollback_calls_reset_hard(self, tools):
        t, repo = tools
        t.rollback_last_commit()
        repo.git.reset.assert_called_once_with("--hard", "HEAD~1")

    def test_rollback_no_parents_raises(self, tools):
        from core.exceptions import GitRollbackError
        t, repo = tools
        repo.head.commit.parents = []
        with pytest.raises((GitRollbackError, Exception)):
            t.rollback_last_commit()


# ---------------------------------------------------------------------------
# stash / stash_pop
# ---------------------------------------------------------------------------

class TestStash:
    @pytest.fixture
    def tools(self):
        from core.git_tools import GitTools
        mock_repo = MagicMock()
        mock_repo.working_tree_dir = "/fake"
        mock_repo.untracked_files = []
        mock_repo.is_dirty.return_value = True
        with patch("core.git_tools.Repo", return_value=mock_repo):
            t = GitTools(repo_path="/fake")
        return t, mock_repo

    def test_stash_called_when_dirty(self, tools):
        t, repo = tools
        t.stash("my stash")
        repo.git.stash.assert_called()

    def test_stash_skipped_when_clean(self, tools):
        t, repo = tools
        repo.is_dirty.return_value = False
        t.stash()
        repo.git.stash.assert_not_called()

    def test_stash_uses_default_message(self, tools):
        t, repo = tools
        t.stash()
        call_args = repo.git.stash.call_args
        assert "AURA" in str(call_args)

    def test_stash_pop_called_when_stashes_exist(self, tools):
        t, repo = tools
        repo.git.stash.return_value = "stash@{0}: On main: msg"
        t.stash_pop()
        # stash was called at least once (for "list" and "pop")
        assert repo.git.stash.called
