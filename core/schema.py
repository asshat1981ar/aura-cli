"""Schema validation, typed LLM parsing, and routing primitives for AURA."""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Typed LLM output parser
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

T = TypeVar("T", bound=BaseModel)


def parse_llm_to_model(raw_response: str, model_class: Type[T]) -> T:
    """Extract a JSON object from a raw LLM string and parse it into *model_class*.

    Handles three common LLM output formats:
    1. Fenced code block: ```json { ... } ```
    2. Bare JSON object embedded in prose (first ``{`` … last ``}``)
    3. The raw string itself when it is already valid JSON

    Raises:
        ValueError: When the response cannot be parsed or fails Pydantic validation.
    """
    json_str = raw_response
    match = _JSON_BLOCK_RE.search(raw_response)
    if match:
        json_str = match.group(1)
    else:
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        if start != -1 and end > start:
            json_str = raw_response[start:end]

    try:
        parsed_dict = json.loads(json_str)
        return model_class(**parsed_dict)
    except (json.JSONDecodeError, ValidationError) as exc:
        log_json(
            "ERROR",
            "parse_llm_to_model_failed",
            details={"model": model_class.__name__, "error": str(exc)},
        )
        raise ValueError(f"LLM Output Schema Violation ({model_class.__name__}): {exc}") from exc


# ---------------------------------------------------------------------------
# Routing decision enum
# ---------------------------------------------------------------------------


class RoutingDecision(str, Enum):
    """Typed routing outcome returned by ``LoopOrchestrator._route_failure()``.

    Using a ``str`` enum keeps it backward-compatible with any existing code
    that compares against plain strings (``route == "act"`` still works).
    """

    ACT = "act"    # Recoverable code-level error — retry the act phase
    PLAN = "plan"  # Structural / design error — re-plan from scratch
    SKIP = "skip"  # External / environment issue — cannot be self-healed


# ---------------------------------------------------------------------------
# Phase schema registry
# ---------------------------------------------------------------------------

PHASE_SCHEMA = {
    "context": {"required": ["goal", "snapshot", "memory_summary", "constraints"]},
    "plan": {"required": ["steps", "risks"]},
    "critique": {"required": ["issues", "fixes"]},
    "task_bundle": {"required": ["tasks"]},
    "change_set": {"required": ["changes"]},
    "verification": {"required": ["status", "failures", "logs"]},
    "reflection": {"required": ["summary", "learnings", "next_actions"]},
}


def validate_phase_output(name: str, payload: dict) -> list[str]:
    schema = PHASE_SCHEMA.get(name)
    if not schema:
        return [f"Unknown phase '{name}'"]
    missing = [key for key in schema.get("required", []) if key not in payload]
    return [f"Missing key: {key}" for key in missing]
