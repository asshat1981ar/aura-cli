from __future__ import annotations

from unittest.mock import MagicMock

from core.propagation_engine import PropagationEngine


class _Queue:
    def __init__(self):
        self.items: list[str] = []

    def add(self, goal: str) -> None:
        self.items.append(goal)

    def _load_queue(self):
        return list(self.items)


def _entry_with_remediation(*, route: str, repeated: bool, signals: list[str], goal: str = "Fix parser"):
    return {
        "cycle_id": "cycle-1",
        "goal": goal,
        "goal_type": "bug_fix",
        "phase_outputs": {
            "verification": {
                "status": "fail",
                "failures": ["SyntaxError: invalid syntax"],
                "logs": "SyntaxError: invalid syntax",
                "failure_investigation": {
                    "signals": signals,
                    "repeated_failure_detected": repeated,
                },
                "remediation_plan": {
                    "route": route,
                    "repeated_failure_detected": repeated,
                    "next_checks": ["Inspect the generated file."],
                },
            },
            "apply_result": {"applied": [], "failed": []},
            "reflection": {"learnings": []},
        },
    }


def test_repeated_failure_queues_follow_up_goal():
    queue = _Queue()
    memory = MagicMock()
    memory.query.return_value = []

    engine = PropagationEngine(queue, context_graph=None, memory_store=memory)
    queued = engine.on_cycle_complete(
        _entry_with_remediation(route="replan", repeated=True, signals=["syntax_error"])
    )

    assert len(queued) == 1
    assert any("Investigate repeated failure" in goal for goal in queued)
    memory.put.assert_called()


def test_skip_route_queues_external_blocker_goal():
    queue = _Queue()
    memory = MagicMock()
    memory.query.return_value = []

    engine = PropagationEngine(queue, context_graph=None, memory_store=memory)
    queued = engine.on_cycle_complete(
        _entry_with_remediation(route="skip", repeated=False, signals=["environment"], goal="Sync secrets")
    )

    assert any("Review external blocker" in goal for goal in queued)


def test_one_off_act_retry_does_not_queue_remediation_goal():
    queue = _Queue()
    memory = MagicMock()
    memory.query.return_value = []

    engine = PropagationEngine(queue, context_graph=None, memory_store=memory)
    queued = engine.on_cycle_complete(
        _entry_with_remediation(route="retry", repeated=False, signals=["assertion_failure"])
    )

    assert queued == []
