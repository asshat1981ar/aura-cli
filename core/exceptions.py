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

# Typed execution/validation hierarchy
class ValidationError(AuraError):
    """Raised when input validation fails."""
    pass

class PhaseValidationError(ValidationError):
    """Raised when a pipeline phase emits invalid or missing fields."""
    def __init__(self, phase: str, message: str):
        super().__init__(f"{phase}: {message}")
        self.phase = phase

class PhaseExecutionError(AuraError):
    """Raised when a pipeline phase fails unexpectedly."""
    def __init__(self, phase: str, message: str):
        super().__init__(f"{phase}: {message}")
        self.phase = phase

class SkillExecutionError(PhaseExecutionError):
    """Raised when a skill fails during dispatch."""
    pass

class SandboxExecutionError(PhaseExecutionError):
    """Raised when sandbox execution fails."""
    pass

class VerificationError(PhaseExecutionError):
    """Raised when verification fails unexpectedly."""
    pass

class PaginationError(AuraError):
    """Raised when pagination/filtering parameters are invalid."""
    pass

class PreflightError(ValidationError):
    """Raised when known-bad requests are short-circuited."""
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
