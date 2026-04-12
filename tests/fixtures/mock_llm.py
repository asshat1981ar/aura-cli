"""Deterministic mock LLM harness for the AURA test suite.

Provides ``MockModelAdapter`` — a drop-in replacement for
``core.model_adapter.ModelAdapter`` that never makes real API calls.
Responses are resolved by substring-matching the incoming prompt against a
registry of ``pattern → response_text`` pairs supplied at construction time
or registered later via ``set_response``.
"""

from __future__ import annotations

from typing import Any

from tests.fixtures.mock_responses import (
    DEFAULT_RESPONSE,
    CODER_RESPONSE,
    CRITIC_RESPONSE,
    PLANNER_RESPONSE,
    REFLECTOR_RESPONSE,
)

# Default pattern registry — covers the four canonical pipeline phases.
_DEFAULT_PATTERNS: dict[str, str] = {
    "plan": PLANNER_RESPONSE,
    "code": CODER_RESPONSE,
    "critique": CRITIC_RESPONSE,
    "reflect": REFLECTOR_RESPONSE,
}


class MockModelAdapter:
    """Deterministic mock for ModelAdapter.

    Returns fixed responses by prompt pattern so tests never require an API
    key or network access.

    Parameters
    ----------
    responses:
        Optional mapping of *substring pattern* → *response text*.  If a
        prompt contains the pattern (case-insensitive) the associated text is
        returned.  When multiple patterns match the *longest* pattern wins,
        giving more specific rules higher priority.  If *responses* is ``None``
        the built-in defaults from ``mock_responses`` are used.
    """

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        # Use caller-supplied mapping or fall back to the built-in defaults.
        self._responses: dict[str, str] = dict(responses) if responses is not None else dict(_DEFAULT_PATTERNS)
        # Track every call for assertion convenience.
        self.call_log: list[str] = []

    # ------------------------------------------------------------------
    # Public mutation API
    # ------------------------------------------------------------------

    def set_response(self, pattern: str, response: str) -> None:
        """Register (or override) a pattern → response mapping.

        Parameters
        ----------
        pattern:
            A substring that will be searched for in incoming prompts
            (case-insensitive).
        response:
            The text to return when *pattern* matches.
        """
        self._responses[pattern] = response

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:  # noqa: ARG002
        """Return a deterministic response for *prompt*.

        Matches the prompt against the registered patterns (longest match wins)
        and returns the corresponding response text.  Falls back to
        ``DEFAULT_RESPONSE`` when nothing matches.
        """
        self.call_log.append(prompt)
        return self._match(prompt)

    def respond(self, prompt: str) -> str:
        """Alias for :meth:`generate` matching the ``ModelAdapter.respond`` signature."""
        return self.generate(prompt)

    def respond_for_role(self, route_key: str, prompt: str) -> str:  # noqa: ARG002
        """Alias for :meth:`generate` matching the ``ModelAdapter.respond_for_role`` signature.

        *route_key* is accepted but ignored; the response is determined solely
        by the *prompt* content.
        """
        return self.generate(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match(self, prompt: str) -> str:
        """Return the response for the longest pattern found in *prompt*.

        Case-insensitive substring matching is used so patterns remain readable
        while staying robust to prompt phrasing variations.
        """
        lower_prompt = prompt.lower()
        # Sort patterns longest-first so more specific entries win over generic
        # ones (e.g. "plan and code" beats "plan").
        matched_patterns = [p for p in self._responses if p.lower() in lower_prompt]
        if not matched_patterns:
            return DEFAULT_RESPONSE
        best = max(matched_patterns, key=len)
        return self._responses[best]
