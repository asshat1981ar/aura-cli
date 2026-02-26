from typing import List, Dict


def _count_list(value) -> int:
    if isinstance(value, list):
        return len(value)
    return 0


def format_decision_log(history: List[Dict]) -> str:
    lines = []
    for entry in history:
        phase_outputs = entry.get("phase_outputs", {})
        plan = phase_outputs.get("plan", {})
        critique = phase_outputs.get("critique", {})
        task_bundle = phase_outputs.get("task_bundle", {})
        change_set = phase_outputs.get("change_set", {})
        verification = phase_outputs.get("verification", {})
        reflection = phase_outputs.get("reflection", {})

        steps = _count_list(plan.get("steps"))
        issues = _count_list(critique.get("issues"))
        tasks = _count_list(task_bundle.get("tasks"))
        changes = _count_list(change_set.get("changes"))

        lines.append(f"Cycle: {entry.get('cycle_id', 'unknown')}")
        lines.append(f"  Plan steps: {steps}")
        lines.append(f"  Critique issues: {issues}")
        lines.append(f"  Tasks: {tasks}")
        lines.append(f"  Changes: {changes}")
        lines.append(f"  Verification: {verification.get('status', 'unknown')}")
        lines.append(f"  Reflection: {reflection.get('summary', '')}")
        if entry.get("stop_reason"):
            lines.append(f"  Stop reason: {entry['stop_reason']}")
        lines.append("")
    return "\n".join(lines).strip()
