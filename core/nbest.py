"""N-Best code generation with critic tournament scoring.

Generates N code variants with temperature variation, sandboxes all candidates,
then runs a critic tournament to select the highest-quality implementation.
This replaces single-path code generation with diversity-driven selection.
"""
import json
import re
from dataclasses import dataclass, field

from core.logging_utils import log_json

SCORING_AXES = ["correctness", "elegance", "efficiency", "maintainability", "test_coverage"]


@dataclass
class CodeCandidate:
    """A single code generation candidate with scoring metadata."""
    variant_id: int
    changes: list[dict] = field(default_factory=list)
    temperature: float = 0.5
    raw_response: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    sandbox_passed: bool = False
    sandbox_output: str = ""


class NBestEngine:
    """Generates N code variants and runs critic tournament to select best.

    Usage::

        engine = NBestEngine(n_candidates=3)
        candidates = engine.generate_candidates(model, prompt, context)
        candidates = engine.sandbox_all(sandbox_agent, candidates, context)
        winner = engine.critic_tournament(model, candidates, goal)
    """

    DEFAULT_TEMPERATURES = (0.2, 0.5, 0.8)

    def __init__(self, n_candidates: int = 3,
                 temperature_spread: tuple[float, ...] | None = None):
        self.n_candidates = max(1, n_candidates)
        self.temperature_spread = temperature_spread or self.DEFAULT_TEMPERATURES

    def generate_candidates(self, model, prompt: str,
                            context: dict | None = None) -> list[CodeCandidate]:
        """Generate N code variants with temperature variation."""
        temps = list(self.temperature_spread[:self.n_candidates])
        # Pad if fewer temperatures than candidates
        while len(temps) < self.n_candidates:
            temps.append(temps[-1])

        candidates = []
        for i, temp in enumerate(temps):
            approach = "conservative" if temp < 0.4 else "moderate" if temp < 0.7 else "creative"
            variant_prompt = (
                f"{prompt}\n\n[Variant {i + 1}/{len(temps)}: "
                f"Explore a {approach} approach]"
            )
            candidate = CodeCandidate(variant_id=i, temperature=temp)
            try:
                respond_fn = getattr(model, "respond_for_role", None)
                if callable(respond_fn):
                    response = respond_fn("code_generation", variant_prompt)
                else:
                    response = model.respond(variant_prompt)
                candidate.raw_response = response
                candidate.changes = self._parse_changes(response)
            except Exception as exc:
                log_json("WARN", "nbest_candidate_failed",
                         details={"variant": i, "error": str(exc)})
                candidate.raw_response = f"ERROR: {exc}"
            candidates.append(candidate)

        log_json("INFO", "nbest_candidates_generated",
                 details={"count": len(candidates),
                          "with_changes": sum(1 for c in candidates if c.changes)})
        return candidates

    def sandbox_all(self, sandbox_agent, candidates: list[CodeCandidate],
                    context: dict | None = None) -> list[CodeCandidate]:
        """Run all candidates through sandbox, mark pass/fail."""
        for candidate in candidates:
            if not candidate.changes:
                continue
            try:
                run_fn = getattr(sandbox_agent, "run", None)
                if callable(run_fn):
                    result = run_fn(candidate.changes, context or {})
                else:
                    result = {"success": True, "output": "sandbox_skipped"}
                candidate.sandbox_passed = result.get("success", False)
                candidate.sandbox_output = str(result.get("output", ""))[:2000]
            except Exception as exc:
                candidate.sandbox_output = str(exc)
        return candidates

    def critic_tournament(self, model, candidates: list[CodeCandidate],
                          goal: str) -> CodeCandidate:
        """Have critic score each candidate on multiple axes, return winner."""
        scoreable = [c for c in candidates if c.changes]
        if not scoreable:
            raise ValueError("No candidates produced valid changes")
        if len(scoreable) == 1:
            scoreable[0].total_score = 1.0
            return scoreable[0]

        comparison_prompt = self._build_comparison_prompt(scoreable, goal)
        try:
            respond_fn = getattr(model, "respond_for_role", None)
            if callable(respond_fn):
                scoring_response = respond_fn("critique", comparison_prompt)
            else:
                scoring_response = model.respond(comparison_prompt)
            self._parse_scores(scoring_response, scoreable)
        except Exception:
            # Fallback: prefer sandbox-passing candidates, then lowest temperature
            for c in scoreable:
                c.total_score = (1.0 if c.sandbox_passed else 0.0) + (1.0 - c.temperature)

        winner = max(scoreable, key=lambda c: c.total_score)
        log_json("INFO", "nbest_tournament_winner",
                 details={"winner_variant": winner.variant_id,
                          "score": winner.total_score,
                          "scores": winner.scores,
                          "candidates_scored": len(scoreable)})
        return winner

    def _build_comparison_prompt(self, candidates: list[CodeCandidate],
                                 goal: str) -> str:
        parts = [
            f"You are a code critic. Score each candidate implementation for the goal: {goal}\n",
            f"Score each on these axes (0.0-1.0): {', '.join(SCORING_AXES)}\n",
        ]
        for c in candidates:
            parts.append(
                f"### Candidate {c.variant_id} "
                f"(temp={c.temperature}, sandbox={'PASS' if c.sandbox_passed else 'FAIL'})"
            )
            for change in c.changes[:3]:
                code = str(change.get("new_code", ""))[:500]
                parts.append(f"File: {change.get('file_path', '?')}\n```\n{code}\n```\n")
        parts.append(
            '\nRespond with JSON only: {"scores": {"<variant_id>": '
            '{"correctness": 0.X, "elegance": 0.X, "efficiency": 0.X, '
            '"maintainability": 0.X, "test_coverage": 0.X}}}'
        )
        return "\n".join(parts)

    def _parse_scores(self, response: str, candidates: list[CodeCandidate]):
        """Parse critic scoring response and assign to candidates."""
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            return
        try:
            data = json.loads(json_match.group())
            scores_data = data.get("scores", {})
            for c in candidates:
                variant_scores = scores_data.get(str(c.variant_id), {})
                for axis in SCORING_AXES:
                    c.scores[axis] = float(variant_scores.get(axis, 0.5))
                c.total_score = sum(c.scores.values()) / len(SCORING_AXES)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    def _parse_changes(self, response: str) -> list[dict]:
        """Parse code changes from model response."""
        # Try JSON format first (AURA's standard output)
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                if "changes" in data:
                    return data["changes"]
                if "aura_target" in data and "code" in data:
                    return [{"file_path": data["aura_target"],
                             "old_code": "", "new_code": data["code"]}]
        except (json.JSONDecodeError, TypeError):
            pass

        # Try code block extraction
        blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
        if blocks:
            return [{"file_path": "unknown", "old_code": "", "new_code": blocks[0]}]

        return []
