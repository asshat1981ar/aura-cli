from typing import Dict

from agents.base import Agent


class SynthesizerAgent(Agent):
    name = "synthesize"

    def run(self, input_data: Dict) -> Dict:
        goal = input_data.get("goal", "")
        plan = input_data.get("plan", {})
        critique = input_data.get("critique", {})
        intent = "\n".join(plan.get("steps", []))
        issues = critique.get("issues", [])
        if issues:
            intent = f"{intent}\n\nCritique:\n" + "\n".join(issues)

        files = input_data.get("files")
        tests = input_data.get("tests")
        if not files:
            files = ["core/", "agents/", "memory/"]
        if tests is None:
            tests = ["python3 -m pytest -q"]

        task = {
            "id": "task_1",
            "title": goal or "Unnamed goal",
            "intent": intent.strip() or "No plan provided.",
            "files": files,
            "tests": tests,
        }
        return {"tasks": [task]}
