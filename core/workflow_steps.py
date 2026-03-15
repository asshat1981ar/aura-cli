import time
import traceback
from typing import Any, Dict, Optional

from core.logging_utils import log_json
from core.workflow_models import WorkflowStep, StepResult


def run_skill(skill_name: str, inputs: Dict) -> Dict:
    """Lazily load skill from registry and call it."""
    try:
        from agents.skills.registry import all_skills
        skills = all_skills()
        if skill_name not in skills:
            return {"error": f"Unknown skill '{skill_name}'. Available: {sorted(skills)}"}
        return skills[skill_name].run(inputs)
    except Exception as exc:
        return {"error": f"Skill runner error: {exc}"}


def wire_inputs(
    step: WorkflowStep,
    step_outputs: Dict[str, Dict],
    initial_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the input dict for this step by:
      1. Starting with initial_inputs (e.g. project_root)
      2. Applying static_inputs (override)
      3. Resolving inputs_from wiring ("step.key" references)
    """
    resolved: Dict[str, Any] = {**initial_inputs, **step.static_inputs}
    for dest_key, src_path in step.inputs_from.items():
        parts = src_path.split(".", 1)
        src_step = parts[0]
        src_key = parts[1] if len(parts) > 1 else None
        src_output = step_outputs.get(src_step, {})
        if src_key == "*" or src_key is None:
            resolved.update(src_output)
        else:
            resolved[dest_key] = src_output.get(src_key)
    return resolved


def execute_step(
    step: WorkflowStep,
    step_outputs: Dict[str, Dict],
    initial_inputs: Dict[str, Any],
) -> StepResult:
    """Run one step with retry + timeout logic."""
    # Check skip condition
    if step.skip_if_false:
        parts = step.skip_if_false.split(".", 1)
        src = step_outputs.get(parts[0], {})
        flag = src.get(parts[1]) if len(parts) > 1 else src
        if not flag:
            return StepResult(
                step_name=step.name,
                status="skipped",
                output={},
                attempts=0,
                elapsed_ms=0.0,
            )

    inputs = wire_inputs(step, step_outputs, initial_inputs)
    last_error: Optional[str] = None
    attempt = 0

    for attempt in range(max(1, step.retry.max_attempts)):
        t0 = time.time()
        try:
            if step.skill_name:
                output = run_skill(step.skill_name, inputs)
            elif step.fn:
                output = step.fn(inputs)
            else:
                output = {"error": f"Step '{step.name}' has no skill_name or fn."}

            elapsed = (time.time() - t0) * 1000

            if isinstance(output, dict) and "error" in output:
                last_error = output["error"]
                log_json("WARN", "workflow_step_error", details={
                    "step": step.name, "attempt": attempt + 1, "error": last_error
                })
            else:
                return StepResult(
                    step_name=step.name,
                    status="ok",
                    output=output or {},
                    attempts=attempt + 1,
                    elapsed_ms=elapsed,
                )

        except Exception as exc:
            elapsed = (time.time() - t0) * 1000
            last_error = f"{type(exc).__name__}: {exc}"
            log_json("WARN", "workflow_step_exception", details={
                "step": step.name, "attempt": attempt + 1, "error": last_error,
                "traceback": traceback.format_exc()[-500:],
            })

        # Backoff before retry
        if attempt < step.retry.max_attempts - 1:
            sleep_t = step.retry.sleep_for(attempt)
            log_json("INFO", "workflow_step_retry", details={
                "step": step.name, "attempt": attempt + 1, "sleep_s": sleep_t
            })
            time.sleep(sleep_t)

    return StepResult(
        step_name=step.name,
        status="failed",
        output={"error": last_error},
        attempts=attempt + 1,
        elapsed_ms=(time.time() - t0) * 1000,
        error=last_error,
    )
