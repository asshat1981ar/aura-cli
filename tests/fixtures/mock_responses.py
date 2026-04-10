"""Pre-built deterministic LLM responses for each AURA pipeline phase.

Each constant is a valid string that the real model would plausibly return for
the corresponding phase.  Using these fixtures keeps tests hermetic and ensures
downstream parsers can exercise their full parsing logic.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Catch-all fallback
# ---------------------------------------------------------------------------

DEFAULT_RESPONSE: str = (
    '{"status": "ok", "message": "Mock LLM default response"}'
)

# ---------------------------------------------------------------------------
# Planner phase — valid JSON plan with 3 steps
# ---------------------------------------------------------------------------

PLANNER_RESPONSE: str = """{
  "goal": "Implement the requested feature",
  "steps": [
    {
      "id": "step_1",
      "title": "Analyse existing code",
      "description": "Read the relevant modules and understand the current structure.",
      "files": ["core/model_adapter.py"]
    },
    {
      "id": "step_2",
      "title": "Implement changes",
      "description": "Write the new functionality following existing conventions.",
      "files": ["core/new_feature.py"]
    },
    {
      "id": "step_3",
      "title": "Write tests",
      "description": "Add unit tests to cover the new code paths.",
      "files": ["tests/test_new_feature.py"]
    }
  ],
  "risks": [],
  "estimated_complexity": "low"
}"""

# ---------------------------------------------------------------------------
# Coder phase — Python code block with AURA_TARGET directive
# ---------------------------------------------------------------------------

CODER_RESPONSE: str = """```python
# AURA_TARGET: /tmp/test_output.py

\"\"\"Generated module — mock coder output.\"\"\"
from __future__ import annotations


def hello(name: str = "world") -> str:
    \"\"\"Return a greeting string.\"\"\"
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(hello())
```"""

# ---------------------------------------------------------------------------
# Critic phase — JSON critique with no blocking issues
# ---------------------------------------------------------------------------

CRITIC_RESPONSE: str = """{
  "issues": [],
  "warnings": [
    {
      "severity": "low",
      "message": "Consider adding a module-level docstring.",
      "file": "/tmp/test_output.py",
      "line": null
    }
  ],
  "blocking": false,
  "summary": "No blocking issues found. The implementation looks correct.",
  "approved": true
}"""

# ---------------------------------------------------------------------------
# Reflector phase — JSON reflection summary
# ---------------------------------------------------------------------------

REFLECTOR_RESPONSE: str = """{
  "summary": "The cycle completed successfully with no errors.",
  "learnings": [
    "The existing test infrastructure supports deterministic mocking.",
    "Pattern-based prompt routing reduces test flakiness."
  ],
  "next_actions": [
    "Expand mock response coverage to additional pipeline phases.",
    "Add property-based tests for the prompt-matching logic."
  ],
  "cycle_outcome": "success",
  "confidence": 0.95
}"""
