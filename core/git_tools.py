from git import Repo
from git.exc import InvalidGitRepositoryError, GitCommandError, NoSuchPathError
import os
from core.logging_utils import log_json # Import the new logging utility

# Custom Exception Classes for GitTools
class GitToolsError(Exception):
    """Base exception for GitTools operations."""
    pass

class GitRepoError(GitToolsError):
    """Exception raised when the Git repository is invalid or not found."""
    pass

class GitCommitError(GitToolsError):
    """Exception raised for errors during Git commit operations."""
    pass

class GitRollbackError(GitToolsError):
    """Exception raised for errors during Git rollback operations."""
    pass

class GitDiffError(GitToolsError):
    """Exception raised for errors during Git diff operations."""
    pass

class GitBranchError(GitToolsError):
    """Exception raised for errors during Git branch operations."""
    pass

class GitStashError(GitToolsError):
    """Exception raised for errors during Git stash operations."""
    pass

class GitStashPopError(GitToolsError):
    """Exception raised for errors during Git stash pop operations."""
    pass

class GitTools:
    def __init__(self, repo_path: str = None):
        try:
            if repo_path:
                self.repo = Repo(repo_path, search_parent_directories=True)
            else:
                # fallback to searching from current working dir
                self.repo = Repo(".", search_parent_directories=True)

            self.repo_root = self.repo.working_tree_dir

        except (InvalidGitRepositoryError, NoSuchPathError) as e:
            log_json("ERROR", "git_repo_init_failed", details={"error": str(e), "repo_path": str(repo_path)})
            raise GitRepoError(
                f"Git repository not found. Start AURA inside a repo or pass repo_path. ({e})"
            )
    
    def commit_all(self, message: str):
        """Commits all changes in the repository."""
        try:
            # Check for untracked files and add them
            untracked_files = self.repo.untracked_files
            if untracked_files:
                self.repo.git.add(untracked_files)
                log_json("INFO", "git_add_untracked", details={"untracked_files_count": len(untracked_files)})
            
            # Add all staged changes and commit
            if self.repo.is_dirty(untracked_files=True): # Check if there are any changes (staged or unstaged)
                self.repo.git.add(A=True) # Stage all changes
                self.repo.index.commit(message)
                log_json("INFO", "git_committed", details={"message": message})
            else:
                log_json("INFO", "git_no_changes_to_commit")
        except GitCommandError as e:
            log_json("ERROR", "git_commit_failed", details={"error": str(e), "message": message})
            raise GitCommitError(f"Failed to commit changes: {e}")

    def rollback_last_commit(self, message: str = "Rollback due to AURA error"):
        """Rolls back the last commit."""
        try:
            if self.repo.head.commit.parents:
                self.repo.git.reset('--hard', 'HEAD~1')
                log_json("INFO", "git_rolled_back_last_commit", details={"message": message})
            else:
                log_json("WARN", "git_rollback_failed_no_parents")
                raise GitRollbackError("Cannot rollback: no previous commits.")
        except GitCommandError as e:
            log_json("ERROR", "git_rollback_failed", details={"error": str(e), "message": message})
            raise GitRollbackError(f"Failed to rollback last commit: {e}")

    def stash(self, message: str = None):
        """Stashes current changes."""
        try:
            # Only stash if there are actual changes (staged or unstaged)
            if self.repo.is_dirty(untracked_files=True):
                stash_message = message if message else 'AURA automated stash'
                self.repo.git.stash('save', stash_message)
                log_json("INFO", "git_stashed", details={"message": stash_message})
            else:
                log_json("INFO", "git_no_changes_to_stash")
        except GitCommandError as e:
            log_json("ERROR", "git_stash_failed", details={"error": str(e), "message": message})
            raise GitStashError(f"Failed to stash changes: {e}")

    def stash_pop(self):
        """Applies the last stashed changes and removes the stash entry."""
        try:
            if self.repo.git.stash('list'):  # Check if there are any stashes
                self.repo.git.stash('pop')
                log_json("INFO", "git_stash_popped")
            else:
                log_json("INFO", "git_no_stashes_to_pop")
        except GitCommandError as e:
            log_json("ERROR", "git_stash_pop_failed", details={"error": str(e)})
            raise GitStashPopError(f"Failed to pop stash: {e}")
