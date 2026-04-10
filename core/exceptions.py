"""Standardized exceptions for AURA CLI.

Provides specific exception types for different error categories
to enable precise error handling and reduce bare 'except Exception' usage.

Error Code Taxonomy:
- AURA-0xx: System errors
- AURA-1xx: Configuration errors  
- AURA-2xx: Authentication/Authorization errors
- AURA-3xx: Network/MCP errors
- AURA-4xx: Filesystem errors
- AURA-5xx: Agent errors
- AURA-6xx: Orchestration errors
- AURA-7xx: SADD errors
- AURA-8xx: Memory errors
- AURA-9xx: Security/Validation errors
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# =============================================================================
# Error Registry - Centralized error definitions
# =============================================================================

ERROR_REGISTRY: Dict[str, Dict[str, Any]] = {
    # AURA-0xx: System errors
    "AURA-000": {
        "severity": "critical",
        "user_message": "An unexpected system error occurred",
        "suggestion": "Please check the logs and report this issue if it persists.",
        "category": "system",
    },
    "AURA-001": {
        "severity": "error",
        "user_message": "System resource exhausted",
        "suggestion": "Free up system resources (memory/disk) and try again.",
        "category": "system",
    },
    "AURA-002": {
        "severity": "error", 
        "user_message": "Operation timed out at system level",
        "suggestion": "The operation took too long. Try with a simpler task or increase timeout.",
        "category": "system",
    },
    "AURA-003": {
        "severity": "warning",
        "user_message": "System interrupt received",
        "suggestion": "Operation was cancelled. You can retry when ready.",
        "category": "system",
    },
    
    # AURA-1xx: Configuration errors
    "AURA-100": {
        "severity": "error",
        "user_message": "Configuration file not found",
        "suggestion": "Run 'aura init' to create a default configuration or specify the correct path.",
        "category": "configuration",
    },
    "AURA-101": {
        "severity": "error",
        "user_message": "Configuration validation failed",
        "suggestion": "Check your aura.config.json for invalid values or syntax errors.",
        "category": "configuration",
    },
    "AURA-102": {
        "severity": "error",
        "user_message": "Missing required configuration key",
        "suggestion": "Add the missing key to your configuration file.",
        "category": "configuration",
    },
    "AURA-103": {
        "severity": "warning",
        "user_message": "Configuration deprecation warning",
        "suggestion": "Update your configuration to use the new key names.",
        "category": "configuration",
    },
    "AURA-104": {
        "severity": "error",
        "user_message": "Environment variable not set",
        "suggestion": "Set the required environment variable or add it to your .env file.",
        "category": "configuration",
    },
    "AURA-105": {
        "severity": "error",
        "user_message": "Invalid environment configuration",
        "suggestion": "Check your .env file for conflicting or invalid values.",
        "category": "configuration",
    },
    
    # AURA-2xx: Authentication/Authorization errors
    "AURA-200": {
        "severity": "error",
        "user_message": "Authentication failed",
        "suggestion": "Check your API credentials and ensure they are valid.",
        "category": "authentication",
    },
    "AURA-201": {
        "severity": "error",
        "user_message": "API key invalid or expired",
        "suggestion": "Verify your API key in the configuration or generate a new one.",
        "category": "authentication",
    },
    "AURA-202": {
        "severity": "error",
        "user_message": "Insufficient permissions",
        "suggestion": "Your credentials don't have permission for this operation.",
        "category": "authentication",
    },
    "AURA-203": {
        "severity": "error",
        "user_message": "Token refresh failed",
        "suggestion": "Re-authenticate using 'aura auth login'.",
        "category": "authentication",
    },
    
    # AURA-3xx: Network/MCP errors
    "AURA-300": {
        "severity": "error",
        "user_message": "Network connection failed",
        "suggestion": "Check your internet connection and try again.",
        "category": "network",
    },
    "AURA-301": {
        "severity": "error",
        "user_message": "MCP server connection failed",
        "suggestion": "Ensure the MCP server is running and accessible.",
        "category": "network",
    },
    "AURA-302": {
        "severity": "warning",
        "user_message": "MCP operation timed out",
        "suggestion": "The MCP server is slow or unresponsive. Retry or check server health.",
        "category": "network",
    },
    "AURA-303": {
        "severity": "error",
        "user_message": "MCP protocol error",
        "suggestion": "There was a communication error with the MCP server. Check server logs.",
        "category": "network",
    },
    "AURA-304": {
        "severity": "error",
        "user_message": "MCP server unavailable",
        "suggestion": "The MCP server is down or restarting. Wait a moment and retry.",
        "category": "network",
    },
    "AURA-305": {
        "severity": "error",
        "user_message": "MCP invalid response",
        "suggestion": "The MCP server returned an unexpected response. Check server version compatibility.",
        "category": "network",
    },
    "AURA-306": {
        "severity": "error",
        "user_message": "MCP retry exhausted",
        "suggestion": "Multiple retry attempts failed. Check MCP server health and configuration.",
        "category": "network",
    },
    "AURA-307": {
        "severity": "error",
        "user_message": "Circuit breaker is open",
        "suggestion": "The service is temporarily disabled due to failures. Wait for recovery.",
        "category": "network",
    },
    "AURA-308": {
        "severity": "error",
        "user_message": "Rate limit exceeded",
        "suggestion": "Too many requests. Wait a moment before trying again.",
        "category": "network",
    },
    "AURA-309": {
        "severity": "error",
        "user_message": "DNS resolution failed",
        "suggestion": "Cannot resolve the server address. Check your network configuration.",
        "category": "network",
    },
    "AURA-310": {
        "severity": "error",
        "user_message": "Network error",
        "suggestion": "A network error occurred. Check your connection and try again.",
        "category": "network",
    },
    
    # AURA-4xx: Filesystem errors
    "AURA-400": {
        "severity": "error",
        "user_message": "File not found",
        "suggestion": "Verify the file path exists and is accessible.",
        "category": "filesystem",
    },
    "AURA-401": {
        "severity": "error",
        "user_message": "Permission denied accessing file",
        "suggestion": "Check file permissions or run with appropriate privileges.",
        "category": "filesystem",
    },
    "AURA-402": {
        "severity": "error",
        "user_message": "Path traversal detected",
        "suggestion": "The path contains invalid characters or attempts to escape the project root.",
        "category": "filesystem",
    },
    "AURA-403": {
        "severity": "error",
        "user_message": "Path outside allowed root",
        "suggestion": "The specified path is outside the allowed project directory.",
        "category": "filesystem",
    },
    "AURA-404": {
        "severity": "error",
        "user_message": "Null byte injection detected",
        "suggestion": "The path contains null bytes which may indicate an injection attempt.",
        "category": "filesystem",
    },
    "AURA-405": {
        "severity": "error",
        "user_message": "Disk full",
        "suggestion": "Free up disk space to continue operations.",
        "category": "filesystem",
    },
    "AURA-406": {
        "severity": "error",
        "user_message": "File is locked or in use",
        "suggestion": "Close any applications using this file and try again.",
        "category": "filesystem",
    },
    "AURA-407": {
        "severity": "error",
        "user_message": "File too large",
        "suggestion": "The file exceeds the maximum allowed size. Use a smaller file or increase limits.",
        "category": "filesystem",
    },
    
    # AURA-5xx: Agent errors
    "AURA-500": {
        "severity": "error",
        "user_message": "Agent initialization failed",
        "suggestion": "Check agent configuration and dependencies.",
        "category": "agent",
    },
    "AURA-501": {
        "severity": "error",
        "user_message": "Agent execution failed",
        "suggestion": "The agent encountered an error. Check logs for details.",
        "category": "agent",
    },
    "AURA-502": {
        "severity": "error",
        "user_message": "Agent timed out",
        "suggestion": "The agent took too long to respond. Try a simpler task or increase timeout.",
        "category": "agent",
    },
    "AURA-503": {
        "severity": "error",
        "user_message": "Agent not found",
        "suggestion": "The requested agent is not registered. Check agent name or register it.",
        "category": "agent",
    },
    "AURA-504": {
        "severity": "warning",
        "user_message": "Agent output validation warning",
        "suggestion": "The agent output didn't match expected schema but may still be usable.",
        "category": "agent",
    },
    
    # AURA-6xx: Orchestration errors
    "AURA-600": {
        "severity": "error",
        "user_message": "Phase execution failed",
        "suggestion": "A pipeline phase failed. Check the specific phase logs for details.",
        "category": "orchestration",
    },
    "AURA-601": {
        "severity": "error",
        "user_message": "Maximum cycles exceeded",
        "suggestion": "The pipeline ran for too many cycles. Review your goal or increase the limit.",
        "category": "orchestration",
    },
    "AURA-602": {
        "severity": "error",
        "user_message": "Verification failed",
        "suggestion": "The verification phase did not pass. Review changes and try again.",
        "category": "orchestration",
    },
    "AURA-603": {
        "severity": "error",
        "user_message": "Pipeline dependency error",
        "suggestion": "A pipeline dependency could not be satisfied.",
        "category": "orchestration",
    },
    "AURA-604": {
        "severity": "error",
        "user_message": "Goal queue error",
        "suggestion": "There was an error processing the goal queue. Check queue state.",
        "category": "orchestration",
    },
    
    # AURA-7xx: SADD errors
    "AURA-700": {
        "severity": "error",
        "user_message": "Workstream execution error",
        "suggestion": "A SADD workstream failed. Check workstream configuration and logs.",
        "category": "sadd",
    },
    "AURA-701": {
        "severity": "error",
        "user_message": "Circular dependency detected",
        "suggestion": "Workstreams have circular dependencies. Review and fix the dependency graph.",
        "category": "sadd",
    },
    "AURA-702": {
        "severity": "error",
        "user_message": "SADD session error",
        "suggestion": "The SADD session encountered an error. Try restarting the session.",
        "category": "sadd",
    },
    "AURA-703": {
        "severity": "error",
        "user_message": "Sub-agent coordination failed",
        "suggestion": "Sub-agents could not coordinate. Check network and agent availability.",
        "category": "sadd",
    },
    
    # AURA-8xx: Memory errors
    "AURA-800": {
        "severity": "error",
        "user_message": "Memory recall failed",
        "suggestion": "Could not retrieve information from memory. Check if data exists.",
        "category": "memory",
    },
    "AURA-801": {
        "severity": "error",
        "user_message": "Memory storage failed",
        "suggestion": "Could not store information to memory. Check disk space and permissions.",
        "category": "memory",
    },
    "AURA-802": {
        "severity": "warning",
        "user_message": "Memory corruption detected",
        "suggestion": "Memory data may be corrupted. Consider clearing and rebuilding memory.",
        "category": "memory",
    },
    
    # AURA-9xx: Security/Validation errors
    "AURA-900": {
        "severity": "critical",
        "user_message": "Security violation detected",
        "suggestion": "A potential security issue was detected. Review the operation carefully.",
        "category": "security",
    },
    "AURA-901": {
        "severity": "critical",
        "user_message": "Hardcoded secret detected",
        "suggestion": "Remove hardcoded secrets from your code. Use environment variables instead.",
        "category": "security",
    },
    "AURA-902": {
        "severity": "critical",
        "user_message": "Injection attack detected",
        "suggestion": "Potential injection attack detected. Operation blocked for security.",
        "category": "security",
    },
    "AURA-903": {
        "severity": "error",
        "user_message": "Schema validation failed",
        "suggestion": "The data does not match the expected schema. Check your input format.",
        "category": "validation",
    },
    "AURA-904": {
        "severity": "error",
        "user_message": "Output validation failed",
        "suggestion": "The output does not match the expected format.",
        "category": "validation",
    },
    "AURA-905": {
        "severity": "error",
        "user_message": "Input validation failed",
        "suggestion": "Check your input parameters and try again.",
        "category": "validation",
    },
    
    # AURA-10xx: Git errors
    "AURA-1000": {
        "severity": "error",
        "user_message": "Git repository error",
        "suggestion": "Ensure you are in a valid git repository.",
        "category": "git",
    },
    "AURA-1001": {
        "severity": "error",
        "user_message": "Git commit failed",
        "suggestion": "Check git status and resolve any conflicts or empty commits.",
        "category": "git",
    },
    "AURA-1002": {
        "severity": "error",
        "user_message": "Git rollback failed",
        "suggestion": "Could not rollback changes. Check git status and resolve manually.",
        "category": "git",
    },
    "AURA-1003": {
        "severity": "error",
        "user_message": "Git operation failed",
        "suggestion": "A git operation failed. Check git configuration and repository state.",
        "category": "git",
    },
}


def get_error_info(code: str) -> Dict[str, Any]:
    """Get error information from registry.
    
    Args:
        code: Error code (e.g., "AURA-100")
        
    Returns:
        Error information dict or default unknown error
    """
    return ERROR_REGISTRY.get(code, {
        "severity": "error",
        "user_message": f"Unknown error ({code})",
        "suggestion": "Please check the logs for more information.",
        "category": "unknown",
    })


def get_error_by_category(category: str) -> Dict[str, Dict[str, Any]]:
    """Get all errors in a category.
    
    Args:
        category: Error category (e.g., "network", "filesystem")
        
    Returns:
        Dict of error codes to error info
    """
    return {
        code: info for code, info in ERROR_REGISTRY.items()
        if info.get("category") == category
    }


# =============================================================================
# Base Exception Classes
# =============================================================================


class AuraCLIError(Exception):
    """Base exception for all AURA CLI errors with error codes.
    
    This is the new error base class that provides rich error information
    including error codes, context, and user-friendly messages.
    
    Args:
        code: Error code from ERROR_REGISTRY (e.g., "AURA-100")
        message: Override message (defaults to registry user_message)
        context: Additional context for the error
        cause: Original exception that caused this error
    """
    
    def __init__(
        self,
        code: str = "AURA-000",
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        self.code = code
        self.error_info = get_error_info(code)
        self.message = message or self.error_info.get("user_message", "Unknown error")
        self.context = context or {}
        self.cause = cause
        self.severity = self.error_info.get("severity", "error")
        self.category = self.error_info.get("category", "unknown")
        self.suggestion = self.error_info.get("suggestion", "")
        
        # Build full message
        full_message = f"[{code}] {self.message}"
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            full_message += f" | Context: {context_str}"
        if cause:
            full_message += f" | Caused by: {type(cause).__name__}: {cause}"
            
        super().__init__(full_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "category": self.category,
            "suggestion": self.suggestion,
            "context": self.context,
            "cause_type": type(self.cause).__name__ if self.cause else None,
            "cause_message": str(self.cause) if self.cause else None,
        }
    
    def __str__(self) -> str:
        return self.message


class AURAError(Exception):
    """Base exception for all AURA errors (legacy, kept for backward compatibility)."""
    pass


# Alias for backward compatibility
AuraError = AURAError
AuraCLIError = AuraCLIError  # Re-export for clarity


# =============================================================================
# File/IO Exceptions (AURA-4xx)
# =============================================================================

class FileToolsError(AURAError):
    """Base exception for file tool operations."""
    pass


class PathTraversalError(FileToolsError):
    """Attempted path traversal attack detected."""
    
    def __init__(self, message: str = "", path: str = ""):
        self.path = path
        super().__init__(message or f"Path traversal detected: {path}")


class PathOutsideRootError(FileToolsError):
    """Path is outside the allowed project root."""
    
    def __init__(self, message: str = "", path: str = "", root: str = ""):
        self.path = path
        self.root = root
        super().__init__(message or f"Path {path} is outside allowed root {root}")


class NullByteInjectionError(FileToolsError):
    """Null byte detected in path (injection attempt)."""
    
    def __init__(self, message: str = "", path: str = ""):
        self.path = path
        super().__init__(message or f"Null byte detected in path: {path}")


class FileNotFoundErrorExtended(FileToolsError):
    """File not found with additional context."""
    
    def __init__(self, message: str = "", path: str = ""):
        self.path = path
        super().__init__(message or f"File not found: {path}")


class FilePermissionError(FileToolsError):
    """Permission denied when accessing file."""
    
    def __init__(self, message: str = "", path: str = ""):
        self.path = path
        super().__init__(message or f"Permission denied: {path}")


class DiskFullError(FileToolsError):
    """Disk is full."""
    pass


class FileLockedError(FileToolsError):
    """File is locked or in use."""
    
    def __init__(self, message: str = "", path: str = ""):
        self.path = path
        super().__init__(message or f"File is locked: {path}")


class FileTooLargeError(FileToolsError):
    """File exceeds maximum size."""
    
    def __init__(self, message: str = "", path: str = "", size: int = 0, max_size: int = 0):
        self.path = path
        self.size = size
        self.max_size = max_size
        super().__init__(
            message or f"File {path} ({size} bytes) exceeds max size ({max_size} bytes)"
        )


# =============================================================================
# MCP/Network Exceptions (AURA-3xx)
# =============================================================================

class MCPError(AURAError):
    """Base exception for MCP-related errors."""
    pass


class MCPConnectionError(MCPError):
    """Failed to connect to MCP server."""
    
    def __init__(self, message: str = "", server: str = "", url: str = ""):
        self.server = server
        self.url = url
        super().__init__(message or f"Failed to connect to MCP server {server} at {url}")


class MCPTimeoutError(MCPError):
    """MCP operation timed out."""
    
    def __init__(self, message: str = "", timeout: float = 0):
        self.timeout = timeout
        super().__init__(message or f"MCP operation timed out after {timeout}s")


class MCPProtocolError(MCPError):
    """MCP protocol error."""
    pass


class MCPServerUnavailableError(MCPError):
    """MCP server unavailable."""
    
    def __init__(self, message: str = "", server: str = ""):
        self.server = server
        super().__init__(message or f"MCP server {server} is unavailable")


class MCPInvalidResponseError(MCPError):
    """MCP invalid response."""
    
    def __init__(self, message: str = "", response: str = ""):
        self.response = response
        super().__init__(message or f"MCP invalid response: {response[:200]}")


class MCPRetryExhaustedError(MCPError):
    """MCP retry exhausted."""
    
    def __init__(self, message: str = "", attempts: int = 0):
        self.attempts = attempts
        super().__init__(message or f"All {attempts} MCP retry attempts failed")


class CircuitBreakerOpenError(MCPError):
    """Circuit breaker is open."""
    
    def __init__(self, message: str = "", circuit_name: str = ""):
        self.circuit_name = circuit_name
        super().__init__(message or f"Circuit breaker {circuit_name} is OPEN")


class RateLimitError(MCPError):
    """Rate limit exceeded."""
    
    def __init__(self, message: str = "", retry_after: float = 0):
        self.retry_after = retry_after
        super().__init__(message or f"Rate limit exceeded. Retry after {retry_after}s")


class DNSResolutionError(MCPError):
    """DNS resolution failed."""
    
    def __init__(self, message: str = "", hostname: str = ""):
        self.hostname = hostname
        super().__init__(message or f"DNS resolution failed for {hostname}")


class NetworkError(MCPError):
    """General network error."""
    pass


# =============================================================================
# Agent Exceptions (AURA-5xx)
# =============================================================================

class AgentError(AURAError):
    """Base exception for agent errors."""
    pass


class AgentInitializationError(AgentError):
    """Failed to initialize agent."""
    
    def __init__(self, message: str = "", agent_name: str = ""):
        self.agent_name = agent_name
        super().__init__(message or f"Failed to initialize agent: {agent_name}")


class AgentExecutionError(AgentError):
    """Agent execution failed."""
    
    def __init__(self, message: str = "", agent_name: str = "", phase: str = ""):
        self.agent_name = agent_name
        self.phase = phase
        super().__init__(message or f"Agent {agent_name} failed in phase {phase}")


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""
    
    def __init__(self, message: str = "", agent_name: str = "", timeout: float = 0):
        self.agent_name = agent_name
        self.timeout = timeout
        super().__init__(message or f"Agent {agent_name} timed out after {timeout}s")


class AgentNotFoundError(AgentError):
    """Agent not found in registry."""
    
    def __init__(self, message: str = "", agent_name: str = ""):
        self.agent_name = agent_name
        super().__init__(message or f"Agent not found: {agent_name}")


class AgentOutputValidationError(AgentError):
    """Agent output validation warning/error."""
    
    def __init__(self, message: str = "", agent_name: str = "", output: str = ""):
        self.agent_name = agent_name
        self.output = output
        super().__init__(message or f"Agent {agent_name} output validation failed")


# =============================================================================
# Orchestration Exceptions (AURA-6xx)
# =============================================================================

class OrchestrationError(AURAError):
    """Base exception for orchestration errors."""
    pass


class PhaseExecutionError(OrchestrationError):
    """Phase execution failed."""
    
    def __init__(self, message: str = "", phase: str = ""):
        self.phase = phase
        super().__init__(message or f"Phase {phase} execution failed")


class CycleExceededError(OrchestrationError):
    """Maximum cycles exceeded."""
    
    def __init__(self, message: str = "", max_cycles: int = 0):
        self.max_cycles = max_cycles
        super().__init__(message or f"Maximum cycles ({max_cycles}) exceeded")


class VerificationFailedError(OrchestrationError):
    """Verification phase failed."""
    
    def __init__(self, message: str = "", verification_type: str = ""):
        self.verification_type = verification_type
        super().__init__(message or f"Verification failed: {verification_type}")


class PipelineDependencyError(OrchestrationError):
    """Pipeline dependency error."""
    pass


class GoalQueueError(OrchestrationError):
    """Goal queue error."""
    pass


# =============================================================================
# SADD Exceptions (AURA-7xx)
# =============================================================================

class SADDError(AURAError):
    """Base exception for SADD-related errors."""
    pass


class WorkstreamError(SADDError):
    """Workstream execution error."""
    
    def __init__(self, message: str = "", workstream_id: str = ""):
        self.workstream_id = workstream_id
        super().__init__(message or f"Workstream {workstream_id} execution failed")


class DependencyCycleError(SADDError):
    """Circular dependency detected in workstreams."""
    
    def __init__(self, message: str = "", cycle: str = ""):
        self.cycle = cycle
        super().__init__(message or f"Circular dependency detected: {cycle}")


class SessionError(SADDError):
    """SADD session error."""
    pass


class SubAgentCoordinationError(SADDError):
    """Sub-agent coordination failed."""
    pass


# =============================================================================
# Memory Exceptions (AURA-8xx)
# =============================================================================

class MemoryError(AURAError):
    """Base exception for memory-related errors."""
    pass


class RecallError(MemoryError):
    """Failed to recall from memory."""
    
    def __init__(self, message: str = "", key: str = ""):
        self.key = key
        super().__init__(message or f"Failed to recall key: {key}")


class StorageError(MemoryError):
    """Failed to store to memory."""
    
    def __init__(self, message: str = "", key: str = ""):
        self.key = key
        super().__init__(message or f"Failed to store key: {key}")


class MemoryCorruptionError(MemoryError):
    """Memory corruption detected."""
    pass


# =============================================================================
# Configuration Exceptions (AURA-1xx)
# =============================================================================

class ConfigError(AURAError):
    """Base exception for configuration errors."""
    pass


class ConfigNotFoundError(ConfigError):
    """Configuration file not found."""
    
    def __init__(self, message: str = "", config_path: str = ""):
        self.config_path = config_path
        super().__init__(message or f"Configuration file not found: {config_path}")


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""
    
    def __init__(self, message: str = "", errors: list = None):
        self.errors = errors or []
        super().__init__(message or f"Configuration validation failed: {errors}")


class ConfigKeyMissingError(ConfigError):
    """Missing required configuration key."""
    
    def __init__(self, message: str = "", key: str = ""):
        self.key = key
        super().__init__(message or f"Missing required configuration key: {key}")


class ConfigDeprecationWarning(ConfigError):
    """Configuration deprecation warning."""
    pass


class EnvironmentVariableError(ConfigError):
    """Environment variable not set or invalid."""
    
    def __init__(self, message: str = "", var_name: str = ""):
        self.var_name = var_name
        super().__init__(message or f"Environment variable not set: {var_name}")


class EnvironmentConfigError(ConfigError):
    """Invalid environment configuration."""
    pass


# Alias for backward compatibility
ConfigurationError = ConfigError


# =============================================================================
# Authentication/Authorization Exceptions (AURA-2xx)
# =============================================================================

class AuthenticationError(AURAError):
    """Base exception for authentication errors."""
    pass


class APIKeyInvalidError(AuthenticationError):
    """API key is invalid or expired."""
    
    def __init__(self, message: str = "", provider: str = ""):
        self.provider = provider
        super().__init__(message or f"API key for {provider} is invalid or expired")


class PermissionDeniedError(AuthenticationError):
    """Insufficient permissions."""
    pass


class TokenRefreshError(AuthenticationError):
    """Token refresh failed."""
    pass


# =============================================================================
# Validation Exceptions (AURA-9xx)
# =============================================================================

class ValidationError(AURAError):
    """Base exception for validation errors."""
    pass


class SchemaValidationError(ValidationError):
    """Schema validation failed."""
    
    def __init__(self, message: str = "", schema_errors: list = None):
        self.schema_errors = schema_errors or []
        super().__init__(message or f"Schema validation failed: {schema_errors}")


class OutputValidationError(ValidationError):
    """Output validation failed."""
    pass


class InputValidationError(ValidationError):
    """Input validation failed."""
    pass


# =============================================================================
# Security Exceptions (AURA-9xx)
# =============================================================================

class SecurityError(AURAError):
    """Base exception for security-related errors."""
    pass


class SecretDetectedError(SecurityError):
    """Hardcoded secret detected."""
    
    def __init__(self, message: str = "", secret_type: str = ""):
        self.secret_type = secret_type
        super().__init__(message or f"Hardcoded {secret_type} detected")


class InjectionDetectedError(SecurityError):
    """Injection attack detected."""
    
    def __init__(self, message: str = "", injection_type: str = ""):
        self.injection_type = injection_type
        super().__init__(message or f"{injection_type} injection detected")


# =============================================================================
# System Exceptions (AURA-0xx)
# =============================================================================

class SystemError(AURAError):
    """Base exception for system errors."""
    pass


class ResourceExhaustedError(SystemError):
    """System resources exhausted."""
    pass


class SystemTimeoutError(SystemError):
    """Operation timed out at system level."""
    pass


class SystemInterruptError(SystemError):
    """System interrupt received (e.g., KeyboardInterrupt)."""
    pass


# =============================================================================
# Git Exceptions (AURA-10xx)
# =============================================================================

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


# =============================================================================
# Error Code Mapping - Maps exception types to error codes
# =============================================================================

ERROR_CODE_MAP = {
    # System errors
    SystemError: "AURA-000",
    ResourceExhaustedError: "AURA-001",
    SystemTimeoutError: "AURA-002",
    SystemInterruptError: "AURA-003",
    
    # Configuration errors
    ConfigNotFoundError: "AURA-100",
    ConfigValidationError: "AURA-101",
    ConfigKeyMissingError: "AURA-102",
    ConfigDeprecationWarning: "AURA-103",
    EnvironmentVariableError: "AURA-104",
    EnvironmentConfigError: "AURA-105",
    
    # Authentication errors
    AuthenticationError: "AURA-200",
    APIKeyInvalidError: "AURA-201",
    PermissionDeniedError: "AURA-202",
    TokenRefreshError: "AURA-203",
    
    # Network/MCP errors
    MCPConnectionError: "AURA-301",
    MCPTimeoutError: "AURA-302",
    MCPProtocolError: "AURA-303",
    MCPServerUnavailableError: "AURA-304",
    MCPInvalidResponseError: "AURA-305",
    MCPRetryExhaustedError: "AURA-306",
    CircuitBreakerOpenError: "AURA-307",
    RateLimitError: "AURA-308",
    DNSResolutionError: "AURA-309",
    NetworkError: "AURA-310",
    
    # Filesystem errors
    FileNotFoundErrorExtended: "AURA-400",
    FilePermissionError: "AURA-401",
    PathTraversalError: "AURA-402",
    PathOutsideRootError: "AURA-403",
    NullByteInjectionError: "AURA-404",
    DiskFullError: "AURA-405",
    FileLockedError: "AURA-406",
    FileTooLargeError: "AURA-407",
    
    # Agent errors
    AgentInitializationError: "AURA-500",
    AgentExecutionError: "AURA-501",
    AgentTimeoutError: "AURA-502",
    AgentNotFoundError: "AURA-503",
    AgentOutputValidationError: "AURA-504",
    
    # Orchestration errors
    PhaseExecutionError: "AURA-600",
    CycleExceededError: "AURA-601",
    VerificationFailedError: "AURA-602",
    PipelineDependencyError: "AURA-603",
    GoalQueueError: "AURA-604",
    
    # SADD errors
    WorkstreamError: "AURA-700",
    DependencyCycleError: "AURA-701",
    SessionError: "AURA-702",
    SubAgentCoordinationError: "AURA-703",
    
    # Memory errors
    RecallError: "AURA-800",
    StorageError: "AURA-801",
    MemoryCorruptionError: "AURA-802",
    
    # Validation errors
    SchemaValidationError: "AURA-903",
    OutputValidationError: "AURA-904",
    InputValidationError: "AURA-905",
    
    # Security errors
    SecretDetectedError: "AURA-901",
    InjectionDetectedError: "AURA-902",
    
    # Git errors
    GitRepoError: "AURA-1000",
    GitCommitError: "AURA-1001",
    GitRollbackError: "AURA-1002",
    GitToolsError: "AURA-1003",
}


def get_error_code_for_exception(exc: Exception) -> str:
    """Get error code for an exception instance.
    
    Args:
        exc: Exception instance
        
    Returns:
        Error code string
    """
    exc_type = type(exc)
    
    # Check for direct match
    if exc_type in ERROR_CODE_MAP:
        return ERROR_CODE_MAP[exc_type]
    
    # Check for AuraCLIError
    if isinstance(exc, AuraCLIError):
        return exc.code
    
    # Check parent classes
    for exc_class, code in ERROR_CODE_MAP.items():
        if isinstance(exc, exc_class):
            return code
    
    # Default unknown error
    return "AURA-000"


def exception_to_aura_cli_error(exc: Exception, context: Dict[str, Any] = None) -> AuraCLIError:
    """Convert any exception to AuraCLIError.
    
    Args:
        exc: Original exception
        context: Additional context
        
    Returns:
        AuraCLIError instance
    """
    if isinstance(exc, AuraCLIError):
        return exc
    
    code = get_error_code_for_exception(exc)
    return AuraCLIError(
        code=code,
        message=str(exc),
        context=context,
        cause=exc,
    )
