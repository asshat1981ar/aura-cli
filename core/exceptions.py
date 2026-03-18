__all__ = [
    "AuraError",
    "GitToolsError", "GitRepoError", "GitCommitError", "GitRollbackError",
    "GitDiffError", "GitBranchError", "GitStashError", "GitStashPopError",
    "GitWorktreeError",
    "GoalQueueError", "BrainError", "ModelAdapterError", "LoopError",
    "ConfigurationError", "SkillError", "SkillTimeoutError",
    "PipelineError", "PhaseError", "EvolutionError", "MutationValidationError",
    "MemoryStoreError", "SubprocessTimeoutError", "SandboxExecutionError",
    "VerificationFailure", "RetryableLLMError",
]


class AuraError(Exception):
    """Base class for all exceptions in AURA."""
    pass

class GitToolsError(AuraError):
    """Base class for all Git-related errors."""
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

class GitWorktreeError(GitToolsError):
    """Exception raised for errors during Git worktree operations."""
    pass

class GoalQueueError(AuraError):
    """Raised when an error occurs in Goal Queue operations."""
    pass

class BrainError(AuraError):
    """Raised when an error occurs in Brain operations."""
    pass

class ModelAdapterError(AuraError):
    """Raised when an error occurs in Model Adapter operations."""
    pass

class LoopError(AuraError):
    """Raised when an error occurs in the Hybrid Loop."""
    pass

class ConfigurationError(AuraError):
    """Raised when there is a configuration-related error."""
    pass


class SkillError(AuraError):
    """Raised when a skill encounters an error."""
    pass


class SkillTimeoutError(SkillError):
    """Raised when a skill execution times out."""
    pass


class PipelineError(AuraError):
    """Raised when the orchestration pipeline encounters an error."""
    pass


class PhaseError(PipelineError):
    """Raised when a specific pipeline phase fails."""
    def __init__(self, phase: str, message: str):
        self.phase = phase
        super().__init__(f"Phase '{phase}' failed: {message}")


class EvolutionError(AuraError):
    """Raised when the evolution loop encounters an error."""
    pass


class MutationValidationError(EvolutionError):
    """Raised when a mutation plan fails validation."""
    pass


class MemoryStoreError(AuraError):
    """Raised when memory store operations fail."""
    pass


class SubprocessTimeoutError(AuraError):
    """Raised when a subprocess exceeds its time limit."""
    pass


class SandboxExecutionError(PhaseError):
    """Raised when sandbox execution fails in a non-recoverable way."""

    def __init__(self, message: str):
        super().__init__("sandbox", message)


class VerificationFailure(PhaseError):
    """Raised when verification reports a failing outcome."""

    def __init__(self, message: str):
        super().__init__("verify", message)


class RetryableLLMError(PhaseError):
    """Raised when an LLM call fails in a way that may succeed on retry."""

    def __init__(self, message: str):
        super().__init__("llm", message)
