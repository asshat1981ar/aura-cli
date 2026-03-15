from typing import Dict, Iterable

from agents.base import Agent


def _normalize_string_list(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    normalized = []
    for item in values:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in normalized:
            continue
        normalized.append(text)
    return normalized


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
        tests = _normalize_string_list(tests)

        beads_context = input_data.get("beads_context", {})
        if not isinstance(beads_context, dict):
            beads_context = {}

        beads_sections = []
        summary = beads_context.get("summary")
        if isinstance(summary, str) and summary.strip():
            beads_sections.append(f"Summary: {summary.strip()}")
        for label, key in (
            ("Required Constraints", "required_constraints"),
            ("Required Skills", "required_skills"),
            ("Follow-up Goals", "follow_up_goals"),
        ):
            values = _normalize_string_list(beads_context.get(key))
            if values:
                beads_sections.append(f"{label}:\n" + "\n".join(f"- {item}" for item in values))
        if beads_sections:
            guidance = "BEADS Guidance:\n" + "\n".join(beads_sections)
            intent = f"{intent}\n\n{guidance}" if intent.strip() else guidance

        required_tests = _normalize_string_list(beads_context.get("required_tests"))
        if required_tests:
            tests = _normalize_string_list([*tests, *required_tests])

        task = {
            "id": "task_1",
            "title": goal or "Unnamed goal",
            "intent": intent.strip() or "No plan provided.",
            "files": files,
            "tests": tests,
        }
        return {"tasks": [task]}
