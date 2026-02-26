from core.policy import Policy


def test_policy_stops_on_max_cycles():
    policy = Policy(max_cycles=2)
    history = [{"phase_outputs": {"verification": {"status": "fail"}}}]
    stop = policy.evaluate(history, {"status": "fail"})
    assert stop == ""

    history.append({"phase_outputs": {"verification": {"status": "fail"}}})
    stop = policy.evaluate(history, {"status": "fail"})
    assert stop == "MAX_CYCLES"


def test_policy_pass():
    policy = Policy(max_cycles=2)
    stop = policy.evaluate([], {"status": "pass"})
    assert stop == "PASS"
