from typing import Dict

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
        return {
            "summary": summary,
            "learnings": learnings,
            "next_actions": input_data.get("next_actions", []),
        }
