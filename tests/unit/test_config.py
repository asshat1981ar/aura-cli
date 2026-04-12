"""Unit tests for configuration management.

Tests cover:
- Configuration validation
- Default values
- Field validation
- Nested configuration
"""

import pytest
from pydantic import ValidationError

from core.config_schema import (
    AuraConfig,
    BeadsConfig,
    ConfigValidator,
    McpServersConfig,
    ModelRoutingConfig,
    SemanticMemoryConfig,
    validate_config,
)


class TestAuraConfigDefaults:
    """Tests for default configuration values."""

    def test_default_config_creation(self):
        """Test creating config with all defaults."""
        config = AuraConfig()

        assert config.model_name == "google/gemini-2.0-flash-exp:free"
        assert config.dry_run is False
        assert config.max_iterations == 10
        assert config.max_cycles == 5
        assert config.policy_name == "sliding_window"

    def test_default_nested_configs(self):
        """Test that nested configs have correct defaults."""
        config = AuraConfig()

        assert isinstance(config.beads, BeadsConfig)
        assert isinstance(config.model_routing, ModelRoutingConfig)
        assert isinstance(config.mcp_servers, McpServersConfig)
        assert isinstance(config.semantic_memory, SemanticMemoryConfig)

    def test_default_beads_config(self):
        """Test beads configuration defaults."""
        beads = BeadsConfig()

        assert beads.enabled is True
        assert beads.required is True
        assert beads.timeout_seconds == 20
        assert beads.scope == "goal_run"

    def test_default_mcp_servers_config(self):
        """Test MCP servers configuration defaults."""
        mcp = McpServersConfig()

        assert mcp.dev_tools == 8001
        assert mcp.skills == 8002
        assert mcp.control == 8003

    def test_default_semantic_memory_config(self):
        """Test semantic memory configuration defaults."""
        memory = SemanticMemoryConfig()

        assert memory.enabled is True
        assert memory.backend == "sqlite_local"
        assert memory.top_k == 10
        assert memory.min_score == 0.65


class TestAuraConfigValidation:
    """Tests for configuration validation."""

    def test_valid_config_with_overrides(self):
        """Test creating config with valid overrides."""
        config = AuraConfig(
            model_name="custom-model",
            max_iterations=20,
            policy_name="fixed",
        )

        assert config.model_name == "custom-model"
        assert config.max_iterations == 20
        assert config.policy_name == "fixed"

    def test_invalid_max_iterations_too_low(self):
        """Test validation fails for max_iterations below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            AuraConfig(max_iterations=0)

        assert "max_iterations" in str(exc_info.value)

    def test_invalid_max_iterations_too_high(self):
        """Test validation fails for max_iterations above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            AuraConfig(max_iterations=1001)

        assert "max_iterations" in str(exc_info.value)

    def test_invalid_max_cycles(self):
        """Test validation fails for invalid max_cycles."""
        with pytest.raises(ValidationError) as exc_info:
            AuraConfig(max_cycles=0)

        assert "max_cycles" in str(exc_info.value)

    def test_invalid_policy_name(self):
        """Test validation fails for invalid policy_name."""
        with pytest.raises(ValidationError) as exc_info:
            AuraConfig(policy_name="INVALID")

        assert "policy_name" in str(exc_info.value)

    def test_invalid_timeout(self):
        """Test validation fails for invalid llm_timeout."""
        with pytest.raises(ValidationError) as exc_info:
            AuraConfig(llm_timeout=0)

        assert "llm_timeout" in str(exc_info.value)

    def test_valid_path_validation(self):
        """Test path field validation accepts valid paths."""
        config = AuraConfig(
            memory_persistence_path="custom/path/tasks.json",
            brain_db_path="custom/path/brain.db",
        )

        assert "custom/path/tasks.json" in config.memory_persistence_path

    def test_empty_path_rejected(self):
        """Test that empty paths are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AuraConfig(memory_persistence_path="")

        assert "memory_persistence_path" in str(exc_info.value)


class TestModelRoutingConfig:
    """Tests for model routing configuration."""

    def test_default_routing(self):
        """Test default model routing values."""
        routing = ModelRoutingConfig()

        assert routing.code_generation == "google/gemini-2.0-flash-exp:free"
        assert routing.planning == "google/gemini-2.0-flash-exp:free"

    def test_routing_with_list(self):
        """Test model routing with list of models."""
        routing = ModelRoutingConfig(
            code_generation=["model1", "model2"],
        )

        assert routing.code_generation == ["model1", "model2"]

    def test_routing_single_model_string(self):
        """Test model routing with single model string."""
        routing = ModelRoutingConfig(planning="custom/planning-model")

        assert routing.planning == "custom/planning-model"


class TestMcpServersConfig:
    """Tests for MCP servers configuration."""

    def test_valid_port_range(self):
        """Test valid port numbers are accepted."""
        mcp = McpServersConfig(
            dev_tools=9000,
            skills=10000,
        )

        assert mcp.dev_tools == 9000

    def test_port_too_low_rejected(self):
        """Test port below 1024 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            McpServersConfig(dev_tools=80)

        assert "dev_tools" in str(exc_info.value)

    def test_port_too_high_rejected(self):
        """Test port above 65535 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            McpServersConfig(dev_tools=70000)

        assert "dev_tools" in str(exc_info.value)


class TestSemanticMemoryConfig:
    """Tests for semantic memory configuration."""

    def test_valid_top_k(self):
        """Test valid top_k values."""
        config = SemanticMemoryConfig(top_k=25)
        assert config.top_k == 25

    def test_top_k_out_of_range(self):
        """Test top_k outside valid range is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticMemoryConfig(top_k=0)

        assert "top_k" in str(exc_info.value)

    def test_min_score_range(self):
        """Test min_score within valid range."""
        config = SemanticMemoryConfig(min_score=0.5)
        assert config.min_score == 0.5

    def test_min_score_out_of_range(self):
        """Test min_score outside valid range is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticMemoryConfig(min_score=1.5)

        assert "min_score" in str(exc_info.value)


class TestConfigValidator:
    """Tests for ConfigValidator class."""

    def test_validate_valid_config(self):
        """Test validating a valid configuration dict."""
        validator = ConfigValidator()
        config_dict = {"model_name": "test-model", "dry_run": True}

        is_valid, validated, errors = validator.validate(config_dict)

        assert is_valid is True
        assert len(errors) == 0
        assert validated["model_name"] == "test-model"

    def test_validate_invalid_config(self):
        """Test validating an invalid configuration dict."""
        validator = ConfigValidator()
        config_dict = {"max_iterations": 0}  # Invalid

        is_valid, validated, errors = validator.validate(config_dict)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_returns_defaults(self):
        """Test that validation fills in defaults."""
        validator = ConfigValidator()
        config_dict = {"model_name": "test"}

        is_valid, validated, errors = validator.validate(config_dict)

        assert is_valid is True
        assert validated["max_iterations"] == 10  # Default value

    def test_get_defaults(self):
        """Test getting default configuration values."""
        validator = ConfigValidator()

        defaults = validator.get_defaults()

        assert "model_name" in defaults
        assert "max_iterations" in defaults


class TestValidateConfigFunction:
    """Tests for validate_config convenience function."""

    def test_validate_config_success(self):
        """Test successful config validation."""
        config_dict = {"model_name": "test-model"}

        is_valid, errors = validate_config(config_dict)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_config_failure(self):
        """Test failed config validation."""
        config_dict = {"policy_name": "INVALID"}

        is_valid, errors = validate_config(config_dict)

        assert is_valid is False
        assert len(errors) > 0


class TestConfigSerialization:
    """Tests for configuration serialization."""

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = AuraConfig(model_name="test-model")

        data = config.model_dump()

        assert data["model_name"] == "test-model"
        assert "beads" in data
        assert isinstance(data["beads"], dict)

    def test_config_json_serialization(self):
        """Test config can be serialized to JSON."""
        import json

        config = AuraConfig()
        data = config.model_dump()

        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_nested_config_serialization(self):
        """Test nested config serialization."""
        config = AuraConfig(
            beads=BeadsConfig(enabled=False, timeout_seconds=30),
        )

        data = config.model_dump()

        assert data["beads"]["enabled"] is False
        assert data["beads"]["timeout_seconds"] == 30
