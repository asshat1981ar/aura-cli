"""Tree-of-Thought planning: generate N plan candidates, score, select best.

Instead of generating a single plan, ToT expands the planning space by:
1. Generating N plans with different strategic prompts (conservative, aggressive, incremental)
2. Self-scoring each plan on feasibility, coverage, risk, and testability
3. Selecting the highest-scoring plan for execution

This dramatically reduces single-path planning failures.
"""
import json
import re
from dataclasses import dataclass, field
from core.logging_utils import log_json

PLAN_STRATEGIES = [
    ("conservative", "Focus on minimal changes. Prefer small, safe modifications. Add thorough error handling."),
    ("aggressive", "Aim for comprehensive solution. Refactor if needed. Prioritize correctness and completeness."),
    ("incremental", "Break into smallest possible steps. Each step should be independently testable."),
    ("test_first", "Start by writing tests that define success. Then implement to pass those tests."),
    ("pattern_based", "Identify similar solved problems. Apply established patterns and best practices."),
]

SCORING_CRITERIA = ["feasibility", "coverage", "risk", "testability", "clarity"]

@dataclass
class PlanCandidate:
    strategy: str
    strategy_description: str
    steps: list = field(default_factory=list)
    raw_response: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    reasoning: str = ""

class TreeOfThoughtPlanner:
    """Generates N plan candidates with varied strategies and picks the best."""

    def __init__(self, n_candidates: int = 3):
        self.n_candidates = min(n_candidates, len(PLAN_STRATEGIES))

    def generate_plans(self, model, goal: str, context: dict) -> list[PlanCandidate]:
        """Generate N plan candidates with different strategic approaches."""
        strategies = PLAN_STRATEGIES[:self.n_candidates]
        candidates = []

        memory_snapshot = context.get("memory_snapshot", "")
        known_weaknesses = context.get("known_weaknesses", "")
        skill_context = context.get("skill_context", {})

        for strategy_name, strategy_desc in strategies:
            candidate = PlanCandidate(strategy=strategy_name, strategy_description=strategy_desc)
            prompt = self._build_plan_prompt(goal, strategy_name, strategy_desc,
                                             memory_snapshot, known_weaknesses, skill_context)
            try:
                respond_fn = getattr(model, "respond_for_role", None)
                if callable(respond_fn):
                    response = respond_fn("planning", prompt)
                else:
                    response = model.respond(prompt)
                candidate.raw_response = response
                candidate.steps = self._parse_steps(response)
            except Exception as exc:
                log_json("WARN", "tot_plan_failed", details={"strategy": strategy_name, "error": str(exc)})
                candidate.raw_response = f"ERROR: {exc}"
            candidates.append(candidate)

        log_json("INFO", "tot_plans_generated",
                 details={"count": len(candidates), "strategies": [c.strategy for c in candidates],
                          "steps_per_plan": [len(c.steps) for c in candidates]})
        return candidates

    def score_plans(self, model, candidates: list[PlanCandidate], goal: str) -> PlanCandidate:
        """Score all plans and return the winner."""
        valid = [c for c in candidates if c.steps]
        if not valid:
            raise ValueError("No valid plan candidates generated")
        if len(valid) == 1:
            valid[0].total_score = 1.0
            return valid[0]

        scoring_prompt = self._build_scoring_prompt(valid, goal)
        try:
            respond_fn = getattr(model, "respond_for_role", None)
            if callable(respond_fn):
                response = respond_fn("analysis", scoring_prompt)
            else:
                response = model.respond(scoring_prompt)
            self._parse_scores(response, valid)
        except (OSError, IOError, ValueError):
            # Fallback: score by step count and strategy preference
            for i, c in enumerate(valid):
                c.total_score = len(c.steps) * 0.1 + (len(valid) - i) * 0.2

        winner = max(valid, key=lambda c: c.total_score)
        log_json("INFO", "tot_plan_winner",
                 details={"strategy": winner.strategy, "score": winner.total_score,
                          "steps": len(winner.steps), "scores": winner.scores})
        return winner

    def _build_plan_prompt(self, goal, strategy_name, strategy_desc, memory, weaknesses, skill_ctx):
        skill_summary = ""
        if isinstance(skill_ctx, dict):
            skill_summary = "\n".join(f"- {k}: {str(v)[:200]}" for k, v in list(skill_ctx.items())[:5])

        return f"""You are an autonomous coding agent planner using the {strategy_name.upper()} strategy.

Strategy: {strategy_desc}

Goal: {goal}

Previous memory: {memory[:1000] if memory else 'None'}
Known weaknesses: {weaknesses[:500] if weaknesses else 'None'}
Skill analysis: {skill_summary or 'None'}

Generate a step-by-step implementation plan. Each step should be:
1. Specific and actionable
2. Include the file(s) to modify
3. Include expected outcome
4. Include how to verify/test

Respond as a JSON array of step objects:
[{{"step": 1, "description": "...", "files": ["..."], "verification": "..."}}]"""

    def _build_scoring_prompt(self, candidates, goal):
        parts = [f"Score each plan for the goal: {goal}\n",
                 f"Criteria (0.0-1.0): {', '.join(SCORING_CRITERIA)}\n"]
        for c in candidates:
            parts.append(f"### Plan: {c.strategy}")
            for i, step in enumerate(c.steps[:5]):
                if isinstance(step, dict):
                    parts.append(f"  {i+1}. {step.get('description', str(step))[:100]}")
                else:
                    parts.append(f"  {i+1}. {str(step)[:100]}")
            parts.append("")
        parts.append('Respond with JSON: {"scores": {"<strategy>": {"feasibility": 0.X, ...}}}')
        return "\n".join(parts)

    def _parse_steps(self, response):
        """Parse plan steps from response."""
        try:
            # Try JSON array
            data = json.loads(response)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "steps" in data:
                return data["steps"]
        except json.JSONDecodeError:
            pass
        # Try numbered list extraction
        steps = []
        for line in response.split("\n"):
            line = line.strip()
            if re.match(r'^\d+[\.\)]\s', line):
                steps.append({"step": len(steps)+1, "description": re.sub(r'^\d+[\.\)]\s*', '', line)})
        return steps if steps else [{"step": 1, "description": response[:500]}]

    def _parse_scores(self, response, candidates):
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            return
        try:
            data = json.loads(json_match.group())
            scores_data = data.get("scores", {})
            for c in candidates:
                strategy_scores = scores_data.get(c.strategy, {})
                for criterion in SCORING_CRITERIA:
                    c.scores[criterion] = float(strategy_scores.get(criterion, 0.5))
                c.total_score = sum(c.scores.values()) / len(SCORING_CRITERIA)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
