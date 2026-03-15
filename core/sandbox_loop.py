"""Sandbox loop logic extracted from orchestrator."""
import time
from typing import Dict, Any, Callable, Tuple, List, Optional
from core.logging_utils import log_json

MAX_SANDBOX_RETRIES = 3

def run_sandbox_loop(
    run_phase: Callable[[str, Dict], Dict],
    notify_ui: Callable[..., None],
    project_root: str,
    goal: str,
    act: Dict,
    task_bundle: Dict,
    dry_run: bool,
    phase_outputs: Dict,
    corr_id: Optional[str] = None,
) -> Tuple[Dict, bool, int]:
    """Run the sandbox pre-apply check, retrying up to MAX_SANDBOX_RETRIES.

    On each failure, injects stderr as a fix_hint and re-generates code.

    Returns:
        Tuple of (final_act_dict, sandbox_passed, act_attempt_delta).
    """
    sandbox_passed = False
    sandbox_result = {}
    act_attempts_used = 0

    for _sandbox_try in range(MAX_SANDBOX_RETRIES):
        notify_ui("on_phase_start", "sandbox")
        t0_sandbox = time.time()
        sandbox_input = {
            "act": act,
            "dry_run": dry_run,
            "project_root": str(project_root),
        }
        if corr_id is not None:
            sandbox_input["corr_id"] = corr_id
        sandbox_result = run_phase("sandbox", sandbox_input) or {}
        notify_ui("on_phase_complete", "sandbox", (time.time() - t0_sandbox) * 1000)

        phase_outputs["sandbox"] = sandbox_result
        sandbox_passed = sandbox_result.get("passed", True)
        if sandbox_passed or dry_run:
            break

        stderr_hint = (
            (sandbox_result.get("details") or {}).get("stderr", "")
            or sandbox_result.get("summary", "sandbox_failed")
        )
        log_json("WARN", "sandbox_pre_apply_failed",
                 details={"try": _sandbox_try + 1,
                          "summary": sandbox_result.get("summary", "")},
                 corr_id=corr_id,
                 phase="sandbox",
                 component="sandbox")

        if _sandbox_try < MAX_SANDBOX_RETRIES - 1:
            task_bundle["fix_hints"] = [stderr_hint]
            act_input = {
                "task": goal,
                "task_bundle": task_bundle,
                "dry_run": dry_run,
                "project_root": str(project_root),
                "fix_hints": [stderr_hint],
            }
            if corr_id is not None:
                act_input["corr_id"] = corr_id
            act = run_phase("act", act_input)
            act_attempts_used += 1
            phase_outputs["change_set"] = act
        else:
            log_json("WARN", "sandbox_max_retries_exceeded",
                     details={"max": MAX_SANDBOX_RETRIES,
                              "continuing_with_best_attempt": True},
                     corr_id=corr_id,
                     phase="sandbox",
                     component="sandbox")

    if not sandbox_passed and not dry_run:
        task_bundle["fix_hints"] = [
            (sandbox_result.get("details") or {}).get("stderr", "")
            or sandbox_result.get("summary", "sandbox_failed")
        ]

    return act, sandbox_passed, act_attempts_used
