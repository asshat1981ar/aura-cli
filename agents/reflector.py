from typing import Dict, List

from agents.base import Agent


class ReflectorAgent(Agent):
    name = "reflect"

    def run(self, input_data: Dict) -> Dict:
        verification = input_data.get("verification", {})
        status = verification.get("status", "skip")
        summary = f"Verification status: {status}."
        failures = verification.get("failures", [])
        
        learnings = []
        if failures:
            learnings.append("Failures: " + "; ".join(failures))
            context_gaps = self._analyze_context_quality(failures)
            if context_gaps:
                learnings.extend(context_gaps)

        return {
            "summary": summary,
            "learnings": learnings,
            "next_actions": input_data.get("next_actions", []),
        }

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
                    break # One gap per failure is enough
        
        return gaps
