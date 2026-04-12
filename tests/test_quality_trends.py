"""Tests for core.quality_trends — quality trend analyzer."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from core.quality_trends import QualitySnapshot, TrendAlert, QualityTrendAnalyzer


class TestQualitySnapshotHealthScore:
    """QualitySnapshot.health_score calculation."""

    def test_perfect_health(self):
        s = QualitySnapshot(verify_status="pass", syntax_errors=0, import_errors=0)
        assert s.health_score == pytest.approx(1.0)

    def test_failed_verify(self):
        s = QualitySnapshot(verify_status="fail", syntax_errors=0, import_errors=0)
        assert s.health_score == pytest.approx(0.4)

    def test_skip_verify(self):
        s = QualitySnapshot(verify_status="skip", syntax_errors=0, import_errors=0)
        assert s.health_score == pytest.approx(0.7)

    def test_syntax_errors_reduce_score(self):
        s = QualitySnapshot(verify_status="pass", syntax_errors=3, import_errors=0)
        # 0.5 + 0.3 - 0.15 + 0.1 = 0.75
        assert s.health_score == pytest.approx(0.75)

    def test_import_errors_reduce_score(self):
        s = QualitySnapshot(verify_status="pass", syntax_errors=0, import_errors=2)
        # 0.5 + 0.3 + 0.1 - 0.10 = 0.80
        assert s.health_score == pytest.approx(0.80)

    def test_many_errors_clamps_to_zero(self):
        s = QualitySnapshot(verify_status="fail", syntax_errors=20, import_errors=20)
        assert s.health_score == 0.0

    def test_health_score_clamped_between_0_and_1(self):
        s = QualitySnapshot(verify_status="pass", syntax_errors=0, import_errors=0)
        assert 0.0 <= s.health_score <= 1.0
        s2 = QualitySnapshot(verify_status="fail", syntax_errors=100, import_errors=100)
        assert 0.0 <= s2.health_score <= 1.0


class TestRecordAndAnalyze:
    """Record snapshots and analyze for alerts."""

    def test_record_returns_empty_for_healthy_snapshot(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        s = QualitySnapshot(verify_status="pass", syntax_errors=0, import_errors=0, test_count=10)
        alerts = analyzer.record(s)
        assert alerts == []
        assert len(analyzer.snapshots) == 1

    def test_record_saves_to_disk(self, tmp_path):
        store = tmp_path / "trends.json"
        analyzer = QualityTrendAnalyzer(store_path=store)
        analyzer.record(QualitySnapshot(verify_status="pass", test_count=5))
        assert store.exists()
        data = json.loads(store.read_text())
        assert len(data["snapshots"]) == 1


class TestThresholdBreachDetection:
    """Threshold breach detection."""

    def test_low_health_score_triggers_alert(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        s = QualitySnapshot(verify_status="fail", syntax_errors=5, import_errors=5)
        alerts = analyzer.record(s)
        types = [a.alert_type for a in alerts]
        assert "threshold_breach" in types

    def test_syntax_errors_above_threshold(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        s = QualitySnapshot(verify_status="pass", syntax_errors=10, import_errors=0)
        alerts = analyzer.record(s)
        syntax_alerts = [a for a in alerts if a.metric == "syntax_errors"]
        assert len(syntax_alerts) == 1
        assert syntax_alerts[0].severity == "critical"

    def test_no_alert_below_threshold(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        s = QualitySnapshot(verify_status="pass", syntax_errors=1, import_errors=0)
        alerts = analyzer.record(s)
        assert all(a.metric != "syntax_errors" for a in alerts)


class TestRegressionDetection:
    """Test count regression detection."""

    def test_test_count_drop_triggers_alert(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        analyzer.record(QualitySnapshot(verify_status="pass", test_count=50))
        alerts = analyzer.record(QualitySnapshot(verify_status="pass", test_count=40))
        regression_alerts = [a for a in alerts if a.alert_type == "regression"]
        assert len(regression_alerts) == 1
        assert regression_alerts[0].metric == "test_count"
        assert regression_alerts[0].severity == "high"
        assert "40" in regression_alerts[0].suggested_goal

    def test_small_test_count_drop_no_alert(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        analyzer.record(QualitySnapshot(verify_status="pass", test_count=50))
        alerts = analyzer.record(QualitySnapshot(verify_status="pass", test_count=48))
        regression_alerts = [a for a in alerts if a.alert_type == "regression"]
        assert len(regression_alerts) == 0

    def test_test_count_increase_no_alert(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        analyzer.record(QualitySnapshot(verify_status="pass", test_count=50))
        alerts = analyzer.record(QualitySnapshot(verify_status="pass", test_count=60))
        regression_alerts = [a for a in alerts if a.alert_type == "regression"]
        assert len(regression_alerts) == 0


class TestConsecutiveFailureDetection:
    """Consecutive failure (degradation) detection."""

    def test_consecutive_failures_trigger_degradation(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        for _ in range(3):
            alerts = analyzer.record(QualitySnapshot(verify_status="fail", syntax_errors=0, import_errors=0))
        degradation = [a for a in alerts if a.alert_type == "degradation"]
        assert len(degradation) == 1
        assert degradation[0].severity == "critical"

    def test_mixed_results_no_degradation(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        analyzer.record(QualitySnapshot(verify_status="fail"))
        analyzer.record(QualitySnapshot(verify_status="pass"))
        alerts = analyzer.record(QualitySnapshot(verify_status="fail"))
        degradation = [a for a in alerts if a.alert_type == "degradation"]
        assert len(degradation) == 0


class TestTrendCalculation:
    """get_trend metric retrieval."""

    def test_health_score_trend(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        analyzer.record(QualitySnapshot(verify_status="pass", syntax_errors=0, import_errors=0))
        analyzer.record(QualitySnapshot(verify_status="fail", syntax_errors=0, import_errors=0))
        trend = analyzer.get_trend("health_score")
        assert len(trend) == 2
        assert trend[0] > trend[1]

    def test_test_count_trend(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        analyzer.record(QualitySnapshot(verify_status="pass", test_count=10))
        analyzer.record(QualitySnapshot(verify_status="pass", test_count=15))
        trend = analyzer.get_trend("test_count")
        assert trend == [10, 15]

    def test_trend_respects_window(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json", window_size=3)
        for i in range(5):
            analyzer.record(QualitySnapshot(verify_status="pass", test_count=i * 10))
        trend = analyzer.get_trend("test_count", window=3)
        assert len(trend) == 3
        assert trend == [20, 30, 40]


class TestSummaryGeneration:
    """get_summary output."""

    def test_empty_summary(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        summary = analyzer.get_summary()
        assert summary["total_cycles"] == 0
        assert summary["health"] == "unknown"

    def test_summary_with_data(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        analyzer.record(QualitySnapshot(verify_status="fail", test_count=5, syntax_errors=0, import_errors=0))
        analyzer.record(QualitySnapshot(verify_status="pass", test_count=10, syntax_errors=0, import_errors=0))
        summary = analyzer.get_summary()
        assert summary["total_cycles"] == 2
        assert summary["window"] == 2
        assert summary["trend"] == "improving"
        assert summary["test_count_current"] == 10
        assert "avg_health" in summary
        assert "current_health" in summary

    def test_declining_trend(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        analyzer.record(QualitySnapshot(verify_status="pass", syntax_errors=0, import_errors=0))
        analyzer.record(QualitySnapshot(verify_status="fail", syntax_errors=0, import_errors=0))
        summary = analyzer.get_summary()
        assert summary["trend"] == "declining"

    def test_stable_trend_single_entry(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        analyzer.record(QualitySnapshot(verify_status="pass"))
        summary = analyzer.get_summary()
        assert summary["trend"] == "stable"


class TestRemediationGoals:
    """get_remediation_goals extraction."""

    def test_goals_from_alerts(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        # Trigger syntax error alert
        analyzer.record(QualitySnapshot(verify_status="pass", syntax_errors=10))
        goals = analyzer.get_remediation_goals()
        assert len(goals) >= 1
        assert any("syntax" in g.lower() for g in goals)

    def test_no_goals_when_healthy(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        analyzer.record(QualitySnapshot(verify_status="pass", syntax_errors=0, import_errors=0, test_count=10))
        goals = analyzer.get_remediation_goals()
        assert goals == []


class TestPersistence:
    """Save and reload from disk."""

    def test_save_and_reload(self, tmp_path):
        store = tmp_path / "trends.json"
        analyzer1 = QualityTrendAnalyzer(store_path=store)
        analyzer1.record(QualitySnapshot(cycle_id="c1", verify_status="pass", test_count=10))
        analyzer1.record(QualitySnapshot(cycle_id="c2", verify_status="fail", syntax_errors=5))

        # Reload from same store
        analyzer2 = QualityTrendAnalyzer(store_path=store)
        assert len(analyzer2.snapshots) == 2
        assert list(analyzer2.snapshots)[0].cycle_id == "c1"
        assert list(analyzer2.snapshots)[1].cycle_id == "c2"

    def test_reload_preserves_alerts(self, tmp_path):
        store = tmp_path / "trends.json"
        analyzer1 = QualityTrendAnalyzer(store_path=store)
        analyzer1.record(QualitySnapshot(verify_status="pass", syntax_errors=10))
        alert_count = len(analyzer1.alerts)
        assert alert_count > 0

        analyzer2 = QualityTrendAnalyzer(store_path=store)
        assert len(analyzer2.alerts) == alert_count

    def test_corrupted_file_handled_gracefully(self, tmp_path):
        store = tmp_path / "trends.json"
        store.write_text("{invalid json!!!")
        analyzer = QualityTrendAnalyzer(store_path=store)
        assert len(analyzer.snapshots) == 0

    def test_missing_file_handled_gracefully(self, tmp_path):
        store = tmp_path / "nonexistent" / "trends.json"
        analyzer = QualityTrendAnalyzer(store_path=store)
        assert len(analyzer.snapshots) == 0


class TestRecordFromCycle:
    """record_from_cycle with mock cycle entry."""

    def test_record_from_cycle_entry(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        cycle_entry = {
            "cycle_id": "cycle-001",
            "goal": "Add authentication module",
            "completed_at": 1700000000.0,
            "duration_s": 45.2,
            "phase_outputs": {
                "quality": {
                    "test_count": 25,
                    "syntax_errors": [],
                    "import_errors": [],
                },
                "verification": {
                    "status": "pass",
                },
                "apply_result": {
                    "applied": ["auth.py", "auth_test.py"],
                },
            },
        }
        alerts = analyzer.record_from_cycle(cycle_entry)
        assert alerts == []
        assert len(analyzer.snapshots) == 1
        snap = list(analyzer.snapshots)[0]
        assert snap.cycle_id == "cycle-001"
        assert snap.goal == "Add authentication module"
        assert snap.test_count == 25
        assert snap.syntax_errors == 0
        assert snap.import_errors == 0
        assert snap.verify_status == "pass"
        assert snap.changes_applied == 2
        assert snap.cycle_duration_s == 45.2

    def test_record_from_cycle_with_errors(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        cycle_entry = {
            "cycle_id": "cycle-002",
            "goal": "Refactor",
            "phase_outputs": {
                "quality": {
                    "test_count": 5,
                    "syntax_errors": ["err1", "err2", "err3", "err4", "err5"],
                    "import_errors": ["imp1"],
                },
                "verification": {
                    "status": "fail",
                },
            },
        }
        alerts = analyzer.record_from_cycle(cycle_entry)
        snap = list(analyzer.snapshots)[0]
        assert snap.syntax_errors == 5
        assert snap.import_errors == 1
        assert snap.verify_status == "fail"
        # Should have threshold breach alerts
        assert len(alerts) > 0

    def test_record_from_empty_cycle(self, tmp_path):
        analyzer = QualityTrendAnalyzer(store_path=tmp_path / "trends.json")
        cycle_entry = {"phase_outputs": {}}
        alerts = analyzer.record_from_cycle(cycle_entry)
        snap = list(analyzer.snapshots)[0]
        assert snap.test_count == 0
        assert snap.verify_status == "skip"
