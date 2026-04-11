# core/agent_sdk/subagent_definitions.py
"""Subagent definitions for parallel task dispatch via Agent SDK.

Each subagent is specialized for a class of work and gets a focused
tool set and system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SubagentDef:
    """Definition for a subagent dispatched by the meta-controller."""

    description: str
    prompt: str
    tools: List[str]
    model: Optional[str] = None


# Task type → subagent name mapping
_TASK_TYPE_MAP: Dict[str, str] = {
    "plan": "planning-agent",
    "planning": "planning-agent",
    "design": "planning-agent",
    "implement": "implementation-agent",
    "code": "implementation-agent",
    "coding": "implementation-agent",
    "build": "implementation-agent",
    "verify": "verification-agent",
    "test": "verification-agent",
    "lint": "verification-agent",
    "security": "verification-agent",
    "research": "research-agent",
    "explore": "research-agent",
    "investigate": "research-agent",
    "analyze": "research-agent",
}


def get_subagent_definitions() -> Dict[str, SubagentDef]:
    """Return all available subagent definitions."""
    return {
        "planning-agent": SubagentDef(
            description=("Senior Software Architect agent for deep planning. Analyzes codebases, identifies risks, decomposes complex goals into ordered implementation steps with dependency tracking."),
            prompt=(
                "You are a Senior Software Architect planning an implementation. "
                "Read the codebase to understand the current architecture. "
                "Produce a detailed, ordered plan with clear steps, file targets, "
                "risk assessments, and verification criteria. "
                "Consider edge cases, backward compatibility, and test strategy."
            ),
            tools=["Read", "Glob", "Grep", "Bash"],
        ),
        "implementation-agent": SubagentDef(
            description=("Expert developer agent for code generation. Writes clean, tested code following project conventions. Handles file creation, editing, and sandbox verification."),
            prompt=(
                "You are an Expert Python Developer implementing a specific task. "
                "Read existing code to match conventions. Write clean code with "
                "type hints. Create or update tests alongside implementation. "
                "Run tests after every change to verify correctness. "
                "Use atomic file operations — never leave partial writes."
            ),
            tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        ),
        "verification-agent": SubagentDef(
            description=("Quality assurance agent for comprehensive verification. Runs tests, linters, type checks, and security scans. Reports findings with severity and fix suggestions."),
            prompt=("You are a Principal QA Engineer verifying code changes. Run the full test suite. Check for lint violations, type errors, and security issues. Report every finding with severity level and a specific fix suggestion. Be thorough — miss nothing."),
            tools=["Read", "Bash", "Glob", "Grep"],
        ),
        "research-agent": SubagentDef(
            description=("Codebase research agent for exploration and context gathering. Explores architecture, traces call chains, identifies patterns, and summarizes findings for other agents."),
            prompt=("You are a codebase researcher gathering context for a development task. Explore the relevant parts of the codebase. Trace call chains, identify patterns, find related code, and summarize your findings concisely. Focus on what's relevant to the task at hand."),
            tools=["Read", "Glob", "Grep", "Bash"],
        ),
    }


def get_subagent_for_task(task_type: str) -> Optional[str]:
    """Return the best subagent name for a task type, or None."""
    return _TASK_TYPE_MAP.get(task_type.lower())
