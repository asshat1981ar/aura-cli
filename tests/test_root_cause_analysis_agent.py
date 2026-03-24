from agents.root_cause_analysis import RootCauseAnalysisAgent


def test_root_cause_analysis_classifies_known_failures():
    agent = RootCauseAnalysisAgent()

    result = agent.run(
        {
            "failures": ["ModuleNotFoundError: No module named 'requests'"],
            "logs": "Traceback ... ModuleNotFoundError: No module named 'requests'",
            "context": {"goal": "Fix import failure", "phase": "verify"},
        }
    )

    assert result["status"] == "analyzed"
    assert "import_error" in result["patterns"]
    assert any("dependency" in cause.lower() or "module" in cause.lower() for cause in result["likely_root_causes"])
    assert result["recommended_actions"]
    assert result["confidence"] >= 0.8


def test_root_cause_analysis_marks_repeated_failures():
    agent = RootCauseAnalysisAgent()
    repeated_entry = {"verification": {"failures": ["AssertionError: expected 1 got 2"]}}

    result = agent.run(
        {
            "failures": ["AssertionError: expected 1 got 2"],
            "logs": "",
            "context": {"phase": "verify"},
            "history": [repeated_entry, repeated_entry],
        }
    )

    assert result["repeated_failure_detected"] is True
    assert any("repeating" in action.lower() for action in result["recommended_actions"])


def test_root_cause_analysis_falls_back_for_unknown_failure():
    agent = RootCauseAnalysisAgent()

    result = agent.run(
        {
            "failures": ["unexpected edge case"],
            "logs": "",
            "context": {},
        }
    )

    assert result["patterns"] == ["unknown_failure"]
    assert result["confidence"] <= 0.5
