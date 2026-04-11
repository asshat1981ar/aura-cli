"""Unit tests for exception handling.

Tests cover:
- Exception class hierarchy
- Error code uniqueness
- Exception attributes and methods
- Error message formatting
"""

import pytest

from core.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentInitializationError,
    AgentTimeoutError,
    AURAError,
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError,
    CycleExceededError,
    DependencyCycleError,
    FileToolsError,
    GitBranchError,
    GitCommitError,
    GitDiffError,
    GitRepoError,
    GitRollbackError,
    GitStashError,
    GitStashPopError,
    GitToolsError,
    InjectionDetectedError,
    MCPConnectionError,
    MCPError,
    MCPInvalidResponseError,
    MCPProtocolError,
    MCPRetryExhaustedError,
    MCPServerUnavailableError,
    MCPTimeoutError,
    MemoryError,
    NullByteInjectionError,
    OrchestrationError,
    OutputValidationError,
    PathOutsideRootError,
    PathTraversalError,
    PhaseExecutionError,
    RecallError,
    SADDError,
    SchemaValidationError,
    SecretDetectedError,
    SecurityError,
    SessionError,
    StorageError,
    ValidationError,
    VerificationFailedError,
    WorkstreamError,
)


class TestAURAErrorBase:
    """Tests for base AURAError class."""

    def test_basic_exception(self):
        """Test basic exception can be raised and caught."""
        with pytest.raises(AURAError):
            raise AURAError("Test error")

    def test_exception_message(self):
        """Test exception message is preserved."""
        with pytest.raises(AURAError, match="Custom message"):
            raise AURAError("Custom message")

    def test_exception_is_exception_subclass(self):
        """Test AURAError is subclass of Exception."""
        assert issubclass(AURAError, Exception)

    def test_exception_str(self):
        """Test exception string representation."""
        err = AURAError("test message")
        assert str(err) == "test message"


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_file_tools_error_hierarchy(self):
        """Test FileToolsError hierarchy."""
        assert issubclass(FileToolsError, AURAError)
        assert issubclass(PathTraversalError, FileToolsError)
        assert issubclass(PathOutsideRootError, FileToolsError)
        assert issubclass(NullByteInjectionError, FileToolsError)

    def test_mcp_error_hierarchy(self):
        """Test MCPError hierarchy."""
        assert issubclass(MCPError, AURAError)
        assert issubclass(MCPConnectionError, MCPError)
        assert issubclass(MCPTimeoutError, MCPError)
        assert issubclass(MCPProtocolError, MCPError)
        assert issubclass(MCPServerUnavailableError, MCPError)
        assert issubclass(MCPInvalidResponseError, MCPError)
        assert issubclass(MCPRetryExhaustedError, MCPError)

    def test_agent_error_hierarchy(self):
        """Test AgentError hierarchy."""
        assert issubclass(AgentError, AURAError)
        assert issubclass(AgentInitializationError, AgentError)
        assert issubclass(AgentExecutionError, AgentError)
        assert issubclass(AgentTimeoutError, AgentError)

    def test_orchestration_error_hierarchy(self):
        """Test OrchestrationError hierarchy."""
        assert issubclass(OrchestrationError, AURAError)
        assert issubclass(PhaseExecutionError, OrchestrationError)
        assert issubclass(CycleExceededError, OrchestrationError)
        assert issubclass(VerificationFailedError, OrchestrationError)

    def test_sadd_error_hierarchy(self):
        """Test SADDError hierarchy."""
        assert issubclass(SADDError, AURAError)
        assert issubclass(WorkstreamError, SADDError)
        assert issubclass(DependencyCycleError, SADDError)
        assert issubclass(SessionError, SADDError)

    def test_memory_error_hierarchy(self):
        """Test MemoryError hierarchy."""
        assert issubclass(MemoryError, AURAError)
        assert issubclass(RecallError, MemoryError)
        assert issubclass(StorageError, MemoryError)

    def test_config_error_hierarchy(self):
        """Test ConfigError hierarchy."""
        assert issubclass(ConfigError, AURAError)
        assert issubclass(ConfigNotFoundError, ConfigError)
        assert issubclass(ConfigValidationError, ConfigError)

    def test_validation_error_hierarchy(self):
        """Test ValidationError hierarchy."""
        assert issubclass(ValidationError, AURAError)
        assert issubclass(SchemaValidationError, ValidationError)
        assert issubclass(OutputValidationError, ValidationError)

    def test_security_error_hierarchy(self):
        """Test SecurityError hierarchy."""
        assert issubclass(SecurityError, AURAError)
        assert issubclass(SecretDetectedError, SecurityError)
        assert issubclass(InjectionDetectedError, SecurityError)

    def test_git_tools_error_hierarchy(self):
        """Test GitToolsError hierarchy."""
        assert issubclass(GitToolsError, AURAError)
        assert issubclass(GitRepoError, GitToolsError)
        assert issubclass(GitCommitError, GitToolsError)
        assert issubclass(GitRollbackError, GitToolsError)
        assert issubclass(GitDiffError, GitToolsError)
        assert issubclass(GitBranchError, GitToolsError)
        assert issubclass(GitStashError, GitToolsError)
        assert issubclass(GitStashPopError, GitStashError)


class TestFileToolsExceptions:
    """Tests for file tools exceptions."""

    def test_path_traversal_error(self):
        """Test PathTraversalError."""
        err = PathTraversalError("Attempted ../ escape")
        assert str(err) == "Attempted ../ escape"

    def test_path_outside_root_error(self):
        """Test PathOutsideRootError."""
        err = PathOutsideRootError("/etc/passwd outside root")
        assert str(err) == "/etc/passwd outside root"

    def test_null_byte_injection_error(self):
        """Test NullByteInjectionError."""
        err = NullByteInjectionError("Null byte in filename")
        assert str(err) == "Null byte in filename"


class TestMCPExceptions:
    """Tests for MCP exceptions."""

    def test_mcp_connection_error(self):
        """Test MCPConnectionError."""
        err = MCPConnectionError("Failed to connect to localhost:8000")
        assert "localhost:8000" in str(err)

    def test_mcp_timeout_error(self):
        """Test MCPTimeoutError."""
        err = MCPTimeoutError("Request timed out after 30s")
        assert "30s" in str(err)

    def test_mcp_server_unavailable(self):
        """Test MCPServerUnavailableError."""
        err = MCPServerUnavailableError("Server dev_tools unavailable")
        assert "dev_tools" in str(err)

    def test_mcp_retry_exhausted(self):
        """Test MCPRetryExhaustedError."""
        err = MCPRetryExhaustedError("All 3 retries failed")
        assert "3 retries" in str(err)


class TestAgentExceptions:
    """Tests for agent exceptions."""

    def test_agent_initialization_error(self):
        """Test AgentInitializationError."""
        err = AgentInitializationError("Failed to load planner agent")
        assert "planner" in str(err)

    def test_agent_execution_error(self):
        """Test AgentExecutionError."""
        err = AgentExecutionError("Agent crashed during execution")
        assert "crashed" in str(err)

    def test_agent_timeout_error(self):
        """Test AgentTimeoutError."""
        err = AgentTimeoutError("Agent exceeded 60s timeout")
        assert "60s" in str(err)


class TestOrchestrationExceptions:
    """Tests for orchestration exceptions."""

    def test_phase_execution_error(self):
        """Test PhaseExecutionError."""
        err = PhaseExecutionError("plan phase failed")
        assert "plan" in str(err)

    def test_cycle_exceeded_error(self):
        """Test CycleExceededError."""
        err = CycleExceededError("Max cycles (5) exceeded")
        assert "5" in str(err)

    def test_verification_failed_error(self):
        """Test VerificationFailedError."""
        err = VerificationFailedError("Output did not match expected")
        assert "Output" in str(err)


class TestConfigExceptions:
    """Tests for configuration exceptions."""

    def test_config_not_found_error(self):
        """Test ConfigNotFoundError."""
        err = ConfigNotFoundError("Config file not found at ~/.aura/config.json")
        assert "~/.aura" in str(err)

    def test_config_validation_error(self):
        """Test ConfigValidationError."""
        err = ConfigValidationError("Invalid log_level: DEBUGG")
        assert "DEBUGG" in str(err)


class TestSecurityExceptions:
    """Tests for security exceptions."""

    def test_secret_detected_error(self):
        """Test SecretDetectedError."""
        err = SecretDetectedError("Hardcoded API key detected")
        assert "API key" in str(err)

    def test_injection_detected_error(self):
        """Test InjectionDetectedError."""
        err = InjectionDetectedError("SQL injection pattern detected")
        assert "SQL" in str(err)


class TestGitToolsExceptions:
    """Tests for git tools exceptions."""

    def test_git_repo_error(self):
        """Test GitRepoError."""
        err = GitRepoError("Not a git repository")
        assert "git repository" in str(err)

    def test_git_commit_error(self):
        """Test GitCommitError."""
        err = GitCommitError("Commit failed: nothing to commit")
        assert "nothing to commit" in str(err)

    def test_git_rollback_error(self):
        """Test GitRollbackError."""
        err = GitRollbackError("Rollback failed: dirty working tree")
        assert "dirty" in str(err)

    def test_git_stash_pop_error(self):
        """Test GitStashPopError."""
        err = GitStashPopError("Stash pop failed: conflicts")
        assert "conflicts" in str(err)


class TestSADDExceptions:
    """Tests for SADD exceptions."""

    def test_workstream_error(self):
        """Test WorkstreamError."""
        err = WorkstreamError("Workstream parsing failed")
        assert "parsing" in str(err)

    def test_dependency_cycle_error(self):
        """Test DependencyCycleError."""
        err = DependencyCycleError("Circular dependency: A -> B -> A")
        assert "Circular" in str(err)

    def test_session_error(self):
        """Test SessionError."""
        err = SessionError("Session expired")
        assert "expired" in str(err)


class TestMemoryExceptions:
    """Tests for memory exceptions."""

    def test_recall_error(self):
        """Test RecallError."""
        err = RecallError("Failed to recall task hierarchy")
        assert "task hierarchy" in str(err)

    def test_storage_error(self):
        """Test StorageError."""
        err = StorageError("Disk full: cannot store memory")
        assert "Disk full" in str(err)


class TestValidationExceptions:
    """Tests for validation exceptions."""

    def test_schema_validation_error(self):
        """Test SchemaValidationError."""
        err = SchemaValidationError("Missing required field: 'goal'")
        assert "goal" in str(err)

    def test_output_validation_error(self):
        """Test OutputValidationError."""
        err = OutputValidationError("Output does not match schema")
        assert "schema" in str(err)


class TestExceptionCatching:
    """Tests for exception catching behavior."""

    def test_catch_specific_with_parent(self):
        """Test catching specific exception with parent class."""
        exceptions = [
            PathTraversalError("test"),
            MCPConnectionError("test"),
            AgentExecutionError("test"),
        ]

        for exc in exceptions:
            try:
                raise exc
            except AURAError as e:
                assert isinstance(e, AURAError)

    def test_catch_parent_not_child(self):
        """Test parent class doesn't catch sibling exceptions."""
        try:
            raise MCPError("test")
        except AgentError:
            pytest.fail("Should not catch MCPError with AgentError")
        except AURAError:
            pass  # This should catch it

    def test_all_exceptions_are_aura_errors(self):
        """Verify all defined exceptions inherit from AURAError."""
        all_exceptions = [
            AURAError,
            FileToolsError,
            PathTraversalError,
            PathOutsideRootError,
            NullByteInjectionError,
            MCPError,
            MCPConnectionError,
            MCPTimeoutError,
            MCPProtocolError,
            MCPServerUnavailableError,
            MCPInvalidResponseError,
            MCPRetryExhaustedError,
            AgentError,
            AgentInitializationError,
            AgentExecutionError,
            AgentTimeoutError,
            OrchestrationError,
            PhaseExecutionError,
            CycleExceededError,
            VerificationFailedError,
            SADDError,
            WorkstreamError,
            DependencyCycleError,
            SessionError,
            MemoryError,
            RecallError,
            StorageError,
            ConfigError,
            ConfigNotFoundError,
            ConfigValidationError,
            ValidationError,
            SchemaValidationError,
            OutputValidationError,
            SecurityError,
            SecretDetectedError,
            InjectionDetectedError,
            GitToolsError,
            GitRepoError,
            GitCommitError,
            GitRollbackError,
            GitDiffError,
            GitStashError,
            GitStashPopError,
        ]

        for exc_class in all_exceptions:
            assert issubclass(exc_class, AURAError), f"{exc_class} is not an AURAError"
