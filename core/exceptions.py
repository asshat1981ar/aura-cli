"""Standardized exceptions for AURA CLI.

Provides specific exception types for different error categories
to enable precise error handling and reduce bare 'except Exception' usage.
"""


class AURAError(Exception):
    """Base exception for all AURA errors."""
    pass


# Alias for backward compatibility
AuraError = AURAError


# File/IO Exceptions
class FileToolsError(AURAError):
    """Base exception for file tool operations."""
    pass


class PathTraversalError(FileToolsError):
    """Attempted path traversal attack detected."""
    pass


class PathOutsideRootError(FileToolsError):
    """Path is outside the allowed project root."""
    pass


class NullByteInjectionError(FileToolsError):
    """Null byte detected in path (injection attempt)."""
    pass


class OldCodeNotFoundError(FileToolsError):
    """Exception raised when old_code is not found in the file."""
    pass


class MismatchOverwriteBlockedError(OldCodeNotFoundError):
    """Raised when mismatch-overwrite fallback is disabled by caller policy."""
    pass


# MCP/Network Exceptions
class MCPError(AURAError):
    """Base exception for MCP-related errors."""
    pass


class MCPConnectionError(MCPError):
    """Failed to connect to MCP server."""
    pass


class MCPTimeoutError(MCPError):
    """MCP operation timed out."""
    pass


class MCPProtocolError(MCPError):
    """MCP protocol error."""
    pass


class MCPServerUnavailableError(MCPError):
    """MCP server unavailable."""
    pass


class MCPInvalidResponseError(MCPError):
    """MCP invalid response."""
    pass


class MCPRetryExhaustedError(MCPError):
    """MCP retry exhausted."""
    pass


# Agent Exceptions
class AgentError(AURAError):
    """Base exception for agent errors."""
    pass


class AgentInitializationError(AgentError):
    """Failed to initialize agent."""
    pass


class AgentExecutionError(AgentError):
    """Agent execution failed."""
    pass


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""
    pass


# Orchestration Exceptions
class OrchestrationError(AURAError):
    """Base exception for orchestration errors."""
    pass


class PhaseExecutionError(OrchestrationError):
    """Phase execution failed."""
    pass


class CycleExceededError(OrchestrationError):
    """Maximum cycles exceeded."""
    pass


class VerificationFailedError(OrchestrationError):
    """Verification phase failed."""
    pass


# SADD Exceptions
class SADDError(AURAError):
    """Base exception for SADD-related errors."""
    pass


class WorkstreamError(SADDError):
    """Workstream execution error."""
    pass


class DependencyCycleError(SADDError):
    """Circular dependency detected in workstreams."""
    pass


class SessionError(SADDError):
    """SADD session error."""
    pass


# Memory Exceptions
class MemoryError(AURAError):
    """Base exception for memory-related errors."""
    pass


class RecallError(MemoryError):
    """Failed to recall from memory."""
    pass


class StorageError(MemoryError):
    """Failed to store to memory."""
    pass


# Configuration Exceptions
class ConfigError(AURAError):
    """Base exception for configuration errors."""
    pass


class ConfigNotFoundError(ConfigError):
    """Configuration file not found."""
    pass


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""
    pass


# Alias for backward compatibility
ConfigurationError = ConfigError


# Validation Exceptions
class ValidationError(AURAError):
    """Base exception for validation errors."""
    pass


class SchemaValidationError(ValidationError):
    """Schema validation failed."""
    pass


class OutputValidationError(ValidationError):
    """Output validation failed."""
    pass


# Security Exceptions
class SecurityError(AURAError):
    """Base exception for security-related errors."""
    pass


class SecretDetectedError(SecurityError):
    """Hardcoded secret detected."""
    pass


class InjectionDetectedError(SecurityError):
    """Injection attack detected."""
    pass


# Git Exceptions
class GitToolsError(AURAError):
    """Base exception for git tool operations."""
    pass


class GitRepoError(GitToolsError):
    """Git repository error."""
    pass


class GitCommitError(GitToolsError):
    """Git commit error."""
    pass


class GitRollbackError(GitToolsError):
    """Git rollback error."""
    pass


class GitDiffError(GitToolsError):
    """Git diff error."""
    pass


class GitBranchError(GitToolsError):
    """Git branch error."""
    pass


class GitStashError(GitToolsError):
    """Git stash error."""
    pass


class GitStashPopError(GitStashError):
    """Git stash pop error."""
    pass
