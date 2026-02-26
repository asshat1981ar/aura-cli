PHASE_SCHEMA = {
    "context": {"required": ["goal", "snapshot", "memory_summary", "constraints"]},
    "plan": {"required": ["steps", "risks"]},
    "critique": {"required": ["issues", "fixes"]},
    "task_bundle": {"required": ["tasks"]},
    "change_set": {"required": ["changes"]},
    "verification": {"required": ["status", "failures", "logs"]},
    "reflection": {"required": ["summary", "learnings", "next_actions"]},
}


def validate_phase_output(name: str, payload: dict) -> list[str]:
    schema = PHASE_SCHEMA.get(name)
    if not schema:
        return [f"Unknown phase '{name}'"]
    missing = [key for key in schema.get("required", []) if key not in payload]
    return [f"Missing key: {key}" for key in missing]
