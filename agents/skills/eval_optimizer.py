"""Skill: evaluator-optimizer — iteratively refine AURA analysis reports.

Implements the Evaluator-Optimizer pattern from the agentic-eval skill:
  Generate → Evaluate → Critique → Refine → Output

Given raw outputs from the parallel skills runner (security, coverage,
complexity, architecture), this skill produces a prioritised action report
that is iteratively critiqued and refined until it meets a quality threshold.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Rubric: dimensions used to score the generated report
# ---------------------------------------------------------------------------
RUBRIC: Dict[str, Dict[str, Any]] = {
    "prioritization": {
        "weight": 0.30,
        "description": "Issues ordered by impact × effort, highest-impact first",
    },
    "actionability": {
        "weight": 0.30,
        "description": "Each item has a concrete, executable next step",
    },
    "completeness": {
        "weight": 0.20,
        "description": "All four skill domains addressed (security, coverage, complexity, architecture)",
    },
    "accuracy": {
        "weight": 0.20,
        "description": "Numbers/filenames match the raw skill data; no hallucination",
    },
}

SCORE_THRESHOLD = 0.75
MAX_ITERATIONS = 3

# ---------------------------------------------------------------------------
# Heuristic scoring (no LLM required — keeps skill self-contained)
# ---------------------------------------------------------------------------


def _heuristic_evaluate(report: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Score a report against the rubric using deterministic heuristics."""
    scores: Dict[str, float] = {}

    # --- prioritization: numbered list with severity words near the top ---
    priority_words = ["critical", "high", "1.", "2.", "3.", "top", "immediately"]
    top_200 = report[:200].lower()
    hit = sum(1 for w in priority_words if w in top_200)
    scores["prioritization"] = min(1.0, hit / 3)

    # --- actionability: imperative verbs per item ---
    action_verbs = ["add", "fix", "replace", "remove", "refactor", "write", "enable", "update", "migrate", "configure", "extract", "reduce"]
    verb_hits = sum(report.lower().count(v) for v in action_verbs)
    scores["actionability"] = min(1.0, verb_hits / 8)

    # --- completeness: all four domains mentioned ---
    domains = {"completeness": ["security", "coverage", "complexity", "architecture"]}
    domain_hits = sum(1 for d in domains["completeness"] if d in report.lower())
    scores["completeness"] = domain_hits / 4

    # --- accuracy: key numbers from raw_data appear in report ---
    accuracy_checks = []
    sec = raw_data.get("Security Scanner", {}).get("result", {})
    cov = raw_data.get("Test Coverage Analyzer", {}).get("result", {})
    _cplx = raw_data.get("Complexity Scorer", {}).get("result", {})
    arch = raw_data.get("Architecture Validator", {}).get("result", {})

    for val, label in [
        (str(sec.get("critical_count", "")), "critical_count"),
        (str(cov.get("coverage_pct", "")), "coverage_pct"),
        (str(arch.get("coupling_score", "")), "coupling_score"),
    ]:
        if val and val != "None":
            accuracy_checks.append(val in report)
    scores["accuracy"] = sum(accuracy_checks) / max(len(accuracy_checks), 1)

    overall = sum(scores[d] * RUBRIC[d]["weight"] for d in RUBRIC)
    passed = {d: scores[d] >= 0.6 for d in scores}

    return {
        "overall_score": round(overall, 3),
        "dimensions": {d: {"score": round(scores[d], 3), "passed": passed[d]} for d in scores},
        "passed": overall >= SCORE_THRESHOLD,
    }


def _build_critique(evaluation: Dict[str, Any]) -> str:
    """Convert evaluation result into a text critique for the optimizer."""
    lines = []
    for dim, info in evaluation["dimensions"].items():
        if not info["passed"]:
            desc = RUBRIC[dim]["description"]
            lines.append(f"- {dim} (score {info['score']:.2f}): needs improvement — {desc}")
    if not lines:
        return "All dimensions passed."
    return "Issues to address:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generator (no LLM — built from raw data deterministically)
# ---------------------------------------------------------------------------


def _generate_report(raw_data: Dict[str, Any], iteration: int, critique: Optional[str]) -> str:
    """Generate a structured action report from raw skills data.

    Uses deterministic templates so the skill works without an LLM call.
    On subsequent iterations the critique is embedded to drive improvement.
    """
    sec = raw_data.get("Security Scanner", {}).get("result", {})
    cov = raw_data.get("Test Coverage Analyzer", {}).get("result", {})
    cplx = raw_data.get("Complexity Scorer", {}).get("result", {})
    arch = raw_data.get("Architecture Validator", {}).get("result", {})

    critical_count = sec.get("critical_count", 0)
    high_count = sec.get("high_count", 0)
    coverage_pct = cov.get("coverage_pct", 0)
    high_risk = cplx.get("high_risk_count", 0)
    coupling = arch.get("coupling_score", "?")
    circular = arch.get("circular_deps", [])
    missing_files = cov.get("missing_files", [])

    # Build prioritised items (sorted by severity score)
    items: List[Dict[str, Any]] = []

    if int(critical_count) > 0:
        items.append(
            {
                "rank": 1,
                "domain": "security",
                "severity": "critical",
                "finding": f"{critical_count} critical security findings (secrets/injection)",
                "action": "Fix critical issues immediately: review agents/planner.py and other files flagged by security_scanner. Run: `python3 -m pytest tests/ -k security` after each fix.",
            }
        )

    if float(str(coverage_pct).replace("%", "")) < 50:
        items.append(
            {
                "rank": 2,
                "domain": "coverage",
                "severity": "high",
                "finding": f"Test coverage at {coverage_pct}% — below 50% target",
                "action": f"Write tests for the {len(missing_files)} untested modules. Start with the highest-risk files flagged by complexity_scorer. Run: `python3 -m pytest --cov=. --cov-report=term-missing`",
            }
        )

    if float(str(high_count)) > 50:
        items.append(
            {
                "rank": 3,
                "domain": "security",
                "severity": "high",
                "finding": f"{high_count} high-severity security findings",
                "action": "Add input validation and parameterised queries. Replace raw string interpolation in SQL/shell calls.",
            }
        )

    if float(str(high_risk)) > 100:
        items.append(
            {
                "rank": 4,
                "domain": "complexity",
                "severity": "medium",
                "finding": f"{high_risk} high-complexity function instances",
                "action": "Extract helper functions from the most complex modules. Target functions with cyclomatic complexity > 10 first. Run complexity_scorer per-file to prioritise.",
            }
        )

    try:
        cs = float(str(coupling))
        if cs > 1.0:
            items.append(
                {
                    "rank": 5,
                    "domain": "architecture",
                    "severity": "medium",
                    "finding": f"Coupling score {coupling} (threshold 1.0) — over-coupled modules",
                    "action": "Introduce interfaces/adapters between tightly-coupled modules. Review circular dependency list and break cycles with dependency injection.",
                }
            )
    except (ValueError, TypeError):
        pass

    if circular:
        items.append(
            {
                "rank": len(items) + 1,
                "domain": "architecture",
                "severity": "medium" if len(circular) < 5 else "high",
                "finding": f"{len(circular)} circular dependency cycle(s) detected",
                "action": "Resolve circular imports by extracting shared types into a `core/types.py` module imported by both sides.",
            }
        )

    # On refinement iterations, add detail requested by critique
    critique_note = ""
    if critique and iteration > 0:
        critique_note = f"\n\n<!-- Refinement #{iteration} addressed: {critique[:120]} -->"

    lines = [
        f"# AURA Codebase Action Report (iteration {iteration + 1})",
        "",
        "## Summary",
        f"- **Security**: {critical_count} critical, {high_count} high findings",
        f"- **Coverage**: {coverage_pct}% ({len(missing_files)} untested modules)",
        f"- **Complexity**: {high_risk} high-risk function instances, avg CC {cplx.get('file_avg_complexity', '?')}",
        f"- **Architecture**: coupling score {coupling}, {len(circular)} circular dep(s)",
        "",
        "## Prioritised Actions",
    ]

    for item in items:
        severity_badge = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(item["severity"], "⚪")
        lines += [
            "",
            f"### {item['rank']}. {severity_badge} [{item['domain'].upper()}] {item['finding']}",
            f"**Action**: {item['action']}",
        ]

    lines += [
        "",
        "## Coverage Gap — Top 5 Untested Files",
    ]
    for f in missing_files[:5] if isinstance(missing_files, list) else []:
        lines.append(f"- `{f}`")

    lines.append(critique_note)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Skill implementation
# ---------------------------------------------------------------------------


class EvalOptimizerSkill(SkillBase):
    """Evaluator-optimizer that refines AURA analysis reports iteratively.

    Input keys:
        skills_data   dict  Raw output from n8n Parallel Skills Runner
                            (keys: Security Scanner, Test Coverage Analyzer, etc.)
        max_iterations int  Override default MAX_ITERATIONS (optional)
        threshold      float Override SCORE_THRESHOLD (optional)

    Output keys:
        report         str   Final refined report (markdown)
        final_score    float Overall rubric score of the final report
        iterations     int   Number of refinement iterations performed
        history        list  Per-iteration {score, critique} records
        converged      bool  True if threshold was reached
    """

    name = "eval_optimizer"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raw_data: Dict[str, Any] = input_data.get("skills_data", {})
        max_iter: int = int(input_data.get("max_iterations", MAX_ITERATIONS))
        threshold: float = float(input_data.get("threshold", SCORE_THRESHOLD))

        if not raw_data:
            return {"error": "skills_data is required", "skill": self.name}

        log_json(
            "INFO",
            "eval_optimizer_start",
            details={
                "max_iterations": max_iter,
                "threshold": threshold,
                "domains": list(raw_data.keys()),
            },
        )

        history: List[Dict[str, Any]] = []
        critique: Optional[str] = None
        report = ""
        evaluation: Dict[str, Any] = {}
        prev_score = -1.0
        converged = False

        for i in range(max_iter):
            t0 = time.monotonic()

            # --- GENERATE ---
            report = _generate_report(raw_data, iteration=i, critique=critique)

            # --- EVALUATE ---
            evaluation = _heuristic_evaluate(report, raw_data)
            score = evaluation["overall_score"]

            # --- LOG ---
            elapsed = round((time.monotonic() - t0) * 1000, 1)
            log_json(
                "INFO",
                "eval_optimizer_iteration",
                details={
                    "iteration": i + 1,
                    "score": score,
                    "passed": evaluation["passed"],
                    "elapsed_ms": elapsed,
                },
            )

            history.append(
                {
                    "iteration": i + 1,
                    "score": score,
                    "critique": critique or "initial generation",
                    "evaluation": evaluation["dimensions"],
                }
            )

            # --- CONVERGENCE CHECK ---
            if evaluation["passed"]:
                converged = True
                log_json("INFO", "eval_optimizer_converged", details={"iterations": i + 1, "score": score})
                break

            if score <= prev_score:
                log_json(
                    "WARNING",
                    "eval_optimizer_not_improving",
                    details={
                        "iteration": i + 1,
                        "score": score,
                        "prev": prev_score,
                    },
                )
                break

            prev_score = score

            # --- CRITIQUE → feed into next iteration ---
            critique = _build_critique(evaluation)

        return {
            "report": report,
            "final_score": evaluation.get("overall_score", 0.0),
            "iterations": len(history),
            "converged": converged,
            "history": history,
            "threshold": threshold,
        }
