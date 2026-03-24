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

class MCPError(AuraError):
    """Base class for all MCP-related errors."""
    pass

class MCPTransportError(MCPError):
    """Raised for errors in the MCP transport layer (e.g., connection failure)."""
    pass

class MCPTimeoutError(MCPTransportError):
    """Raised when an MCP request times out."""
    pass

class MCPServerUnavailableError(MCPTransportError):
    """Raised when an MCP server is unavailable or down."""
    pass

class MCPInvalidResponseError(MCPTransportError):
    """Raised when an MCP server returns a malformed response."""
    pass

class MCPRetryExhaustedError(MCPTransportError):
    """Raised when all retries for an MCP request have failed."""
    pass
