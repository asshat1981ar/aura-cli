from core.investigate_test_drop import investigate_test_count_drop


def test_test_count_drop_summary_and_goal():
    result = investigate_test_count_drop(50, 40, goal="Stabilize suite")

    assert result["delta"] == -10
    assert result["severity"] == "high"
    assert "50 to 40" in result["summary"]
    assert result["suggested_goal"] == "Investigate test count drop for 'Stabilize suite': 50 -> 40"


def test_test_count_drop_to_zero_is_critical():
    result = investigate_test_count_drop(
        12,
        0,
        verification={"status": "fail"},
        remediation_plan={"repeated_failure_detected": True, "route": "replan"},
    )

    assert result["dropped_to_zero"] is True
    assert result["severity"] == "critical"
    assert any("collection outage" in action.lower() for action in result["recommended_actions"])
    assert any("systemic" in cause.lower() for cause in result["likely_causes"])
