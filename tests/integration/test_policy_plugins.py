import time

from core.policy import Policy


def test_sliding_window_policy_stops():
    policy = Policy.from_config({"policy_name": "sliding_window", "policy_max_cycles": 2})
    history = [{"phase_outputs": {"verification": {"status": "fail"}}}]
    stop = policy.evaluate(history, {"status": "fail"})
    assert stop == ""

    history.append({"phase_outputs": {"verification": {"status": "fail"}}})
    stop = policy.evaluate(history, {"status": "fail"})
    assert stop == "MAX_CYCLES"


def test_time_bound_policy_stops():
    policy = Policy.from_config({"policy_name": "time_bound", "policy_max_seconds": 0})
    stop = policy.evaluate([], {"status": "fail"}, started_at=time.time() - 1)
    assert stop == "TIME_LIMIT"
