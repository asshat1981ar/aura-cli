"""A2A Agent Card: capability discovery and advertisement.

An Agent Card is a JSON document served at /.well-known/agent.json that
describes an agent's capabilities, supported protocols, and connection details.
"""
import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AgentCapability:
    """A single capability the agent can perform."""
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCard:
    """A2A Agent Card for capability discovery."""
    name: str = "AURA CLI"
    description: str = "Autonomous software development agent with 10-phase loop"
    version: str = "0.1.0"
    url: str = ""
    capabilities: list[AgentCapability] = field(default_factory=list)
    supported_protocols: list[str] = field(
        default_factory=lambda: ["a2a/1.0", "mcp/1.0"]
    )
    authentication: dict[str, Any] = field(
        default_factory=lambda: {"type": "bearer"}
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls, host: str = "localhost", port: int = 8010) -> "AgentCard":
        """Create default AURA agent card with standard capabilities."""
        return cls(
            url=f"http://{host}:{port}",
            capabilities=[
                AgentCapability(
                    name="code_generation",
                    description="Generate code changes for a given goal",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string"},
                            "context": {"type": "object"},
                        },
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "changes": {"type": "array"},
                            "confidence": {"type": "number"},
                        },
                    },
                ),
                AgentCapability(
                    name="code_review",
                    description="Review code changes and provide critique",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "changes": {"type": "array"},
                            "criteria": {"type": "array"},
                        },
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "score": {"type": "number"},
                            "issues": {"type": "array"},
                        },
                    },
                ),
                AgentCapability(
                    name="test_generation",
                    description="Generate tests for existing code",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "file_paths": {"type": "array"},
                            "framework": {"type": "string"},
                        },
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"test_files": {"type": "array"}},
                    },
                ),
                AgentCapability(
                    name="autonomous_goal",
                    description="Execute full autonomous loop for a goal",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string"},
                            "max_cycles": {"type": "integer"},
                        },
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "changes": {"type": "array"},
                        },
                    },
                ),
                AgentCapability(
                    name="plan_generation",
                    description="Generate implementation plan for a goal",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string"},
                            "codebase_context": {"type": "object"},
                        },
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "steps": {"type": "array"},
                            "estimated_complexity": {"type": "string"},
                        },
                    },
                ),
            ],
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentCard":
        caps_data = data.pop("capabilities", [])
        caps = [AgentCapability(**c) if isinstance(c, dict) else c
                for c in caps_data]
        return cls(
            capabilities=caps,
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
        )
