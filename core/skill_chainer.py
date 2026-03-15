from typing import List

from core.logging_utils import log_json


class SkillChainer:
    """Chains skill results into follow-up goals queued for the next cycle."""

    def maybe_chain(self, skill_name: str, result: dict, goal_queue) -> List[str]:
        """Inspect *result* from *skill_name* and optionally queue remediation goals.

        Currently handles:
        - ``security_scanner``: queues a remediation goal when
          ``critical_count > 0``.

        Args:
            skill_name:  Name of the skill that produced *result*.
            result:      The skill's output dict.
            goal_queue:  A :class:`~core.goal_queue.GoalQueue` instance.

        Returns:
            List of goal strings that were queued (may be empty).
        """
        queued: List[str] = []
        if skill_name == "security_scanner":
            critical_count = result.get("critical_count", 0)
            if critical_count and critical_count > 0:
                goal = (
                    f"Remediate {critical_count} critical security finding(s) "
                    f"identified by the security scanner: "
                    f"{result.get('scan_summary', '')}"
                )
                try:
                    goal_queue.add(goal)
                    queued.append(goal)
                    log_json("INFO", "skill_chainer_queued",
                             details={"skill": skill_name, "goal": goal})
                except Exception as exc:
                    log_json("WARN", "skill_chainer_queue_failed",
                             details={"skill": skill_name, "error": str(exc)})
        return queued


def chain_skill_results(skill_results: dict, goal_queue) -> List[str]:
    """Run :class:`SkillChainer` over all entries in *skill_results*.

    Args:
        skill_results: ``{skill_name: result_dict}`` as returned by
                       :func:`dispatch_skills`.
        goal_queue:    A :class:`~core.goal_queue.GoalQueue` instance.

    Returns:
        Flat list of all goal strings that were queued.
    """
    chainer = SkillChainer()
    all_queued: List[str] = []
    for skill_name, result in skill_results.items():
        all_queued.extend(chainer.maybe_chain(skill_name, result, goal_queue))
    return all_queued
