from agents.investigation_agent import InvestigationAgent


def test_investigation_agent_combines_failure_analysis_and_remediation():
    agent = InvestigationAgent()

    result = agent.run(
        {
            "goal": "Fix parser",
            "verification": {
                "failures": ["SyntaxError: invalid syntax"],
                "logs": "SyntaxError: invalid syntax",
            },
            "context": {"goal": "Fix parser", "phase": "verify", "route": "plan"},
            "route": "plan",
            "root_cause_analysis": {
                "patterns": ["syntax_error"],
                "recommended_actions": ["Inspect the generated file."],
            },
            "history": [
                {"phase_outputs": {"verification": {"failures": ["SyntaxError: invalid syntax"]}}}
            ],
        }
    )

    assert result["status"] == "investigated"
    assert result["verification_investigation"]["repeated_failure_detected"] is True
    assert result["remediation_plan"]["route"] == "replan"
    assert "repeating" in result["summary"]


def test_investigation_agent_includes_test_drop_analysis_when_counts_present():
    agent = InvestigationAgent()

    result = agent.run(
        {
            "goal": "Stabilize suite",
            "verification": {"status": "fail", "failures": ["collection error"], "logs": ""},
            "context": {"goal": "Stabilize suite", "phase": "verify", "route": "plan"},
            "route": "plan",
            "previous_test_count": 25,
            "current_test_count": 0,
        }
    )

    assert result["test_drop_investigation"] is not None
    assert result["test_drop_investigation"]["severity"] == "critical"
    assert "25 to 0" in result["test_drop_investigation"]["summary"]
