from typing import Any, Dict, List

from agents.base import Agent


class ReflectorAgent(Agent):
    name = "reflect"

    def run(self, input_data: Dict) -> Dict:
        verification = input_data.get("verification", {})
        status = verification.get("status", "skip")
        summary = f"Verification status: {status}."
        failures = verification.get("failures", [])
        skill_context: Dict[str, Any] = input_data.get("skill_context", {})

        learnings: List[str] = []
        if failures:
            learnings.append("Failures: " + "; ".join(failures))
            context_gaps = self._analyze_context_quality(failures)
            if context_gaps:
                learnings.extend(context_gaps)

        skill_learnings = self._extract_skill_learnings(skill_context)
        learnings.extend(skill_learnings)

        skill_summary = self._build_skill_summary(skill_context)

        return {
            "summary": summary,
            "learnings": learnings,
            "next_actions": input_data.get("next_actions", []),
            "skill_summary": skill_summary,
            "pipeline_run_id": input_data.get("pipeline_run_id"),
        }

    # ── skill-context helpers ──────────────────────────────────────────────

    def _extract_skill_learnings(self, skill_context: Dict[str, Any]) -> List[str]:
        """Surface actionable signals from skill_dispatch outputs as learnings."""
        learnings: List[str] = []

        security = skill_context.get("security_scanner", {})
        critical = security.get("critical_count", 0)
        if critical:
            learnings.append(f"skill_alert: security_scanner found {critical} critical finding(s)")

        arch = skill_context.get("architecture_validator", {})
        coupling = arch.get("coupling_score")
        if coupling is not None and coupling > 1.0:
            learnings.append(f"skill_alert: architecture coupling {coupling:.2f} exceeds threshold 1.0")

        complexity = skill_context.get("complexity_scorer", {})
        high_risk = complexity.get("high_risk_count", 0)
        if high_risk:
            learnings.append(f"skill_alert: complexity_scorer flagged {high_risk} high-risk function(s)")

        coverage = skill_context.get("test_coverage_analyzer", {})
        if coverage.get("meets_target") is False:
            pct = coverage.get("coverage_pct", 0)
            learnings.append(f"skill_alert: test coverage {pct:.1f}% below target")

        debt = skill_context.get("tech_debt_quantifier", {})
        debt_score = debt.get("debt_score", 0)
        if debt_score > 50:
            learnings.append(f"skill_alert: tech_debt_score {debt_score} is high (>50)")

        return learnings

    def _build_skill_summary(self, skill_context: Dict[str, Any]) -> Dict[str, Any]:
        """Compact summary of skill results for P4 feedback webhook payload."""
        summary: Dict[str, Any] = {}
        extractors: Dict[str, Any] = {
            "security_scanner": lambda d: {"critical": d.get("critical_count", 0), "total": len(d.get("findings", []))},
            "architecture_validator": lambda d: {"coupling": d.get("coupling_score"), "circular_deps": len(d.get("circular_deps", []))},
            "complexity_scorer": lambda d: {"high_risk": d.get("high_risk_count", 0)},
            "test_coverage_analyzer": lambda d: {"coverage_pct": d.get("coverage_pct"), "meets_target": d.get("meets_target")},
            "tech_debt_quantifier": lambda d: {"debt_score": d.get("debt_score")},
            "linter_enforcer": lambda d: {"violations": len(d.get("violations", []))},
        }
        for skill_name, extractor in extractors.items():
            if skill_name in skill_context:
                try:
                    summary[skill_name] = extractor(skill_context[skill_name])
                except Exception:
                    pass
        return summary

    # ── failure analysis ───────────────────────────────────────────────────

    def _analyze_context_quality(self, failures: List[str]) -> List[str]:
        """Detect potential context gaps from failure patterns."""
        gaps = []
        context_signals = [
            ("NameError", "Agent hallucinated a variable or function."),
            ("ImportError", "Agent missed a required dependency."),
            ("ModuleNotFoundError", "Agent hallucinated a module."),
            ("AttributeError", "Agent hallucinated an object attribute."),
            ("not defined", "Agent assumed existence of a symbol."),
        ]

        for f in failures:
            for sig, reason in context_signals:
                if sig in f:
                    gaps.append(f"context_gap: {reason} (Trigger: {sig})")
                    break
        return gaps
