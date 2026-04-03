"""Unit tests for agents/technical_debt_analyzer.py."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from agents.technical_debt_analyzer import TechnicalDebtAnalyzer


class TestTechnicalDebtAnalyzerInit:
    def test_init_sets_project_root(self):
        analyzer = TechnicalDebtAnalyzer("/tmp")
        assert str(analyzer.project_root) == "/tmp"

    def test_init_hotspots_empty(self):
        analyzer = TechnicalDebtAnalyzer(".")
        assert analyzer.hotspots == []


class TestIdentifyHotspots:
    def test_returns_list(self):
        analyzer = TechnicalDebtAnalyzer(".")
        hotspots = analyzer.identify_hotspots()
        assert isinstance(hotspots, list)

    def test_hotspots_have_required_keys(self):
        analyzer = TechnicalDebtAnalyzer(".")
        hotspots = analyzer.identify_hotspots()
        for h in hotspots:
            assert "file_path" in h
            assert "debt_score" in h
            assert "priority" in h

    def test_hotspots_debt_score_is_numeric(self):
        analyzer = TechnicalDebtAnalyzer(".")
        hotspots = analyzer.identify_hotspots()
        for h in hotspots:
            assert isinstance(h["debt_score"], (int, float))

    def test_returns_two_hotspots(self):
        analyzer = TechnicalDebtAnalyzer(".")
        hotspots = analyzer.identify_hotspots()
        assert len(hotspots) == 2


class TestGenerateImprovementPlan:
    def setup_method(self):
        self.analyzer = TechnicalDebtAnalyzer(".")

    def test_plan_has_required_keys(self):
        hotspots = self.analyzer.identify_hotspots()
        plan = self.analyzer.generate_improvement_plan(hotspots)
        assert "steps" in plan
        assert "tools" in plan
        assert "risks" in plan
        assert "success_metrics" in plan

    def test_high_priority_hotspot_adds_steps(self):
        hotspots = [{"file_path": "core/foo.py", "debt_score": 90, "priority": "high", "reasons": []}]
        plan = self.analyzer.generate_improvement_plan(hotspots)
        assert len(plan["steps"]) >= 2
        assert any("core/foo.py" in s for s in plan["steps"])

    def test_medium_priority_hotspot_adds_no_steps(self):
        hotspots = [{"file_path": "core/foo.py", "debt_score": 50, "priority": "medium", "reasons": []}]
        plan = self.analyzer.generate_improvement_plan(hotspots)
        assert plan["steps"] == []

    def test_empty_hotspots_returns_empty_plan(self):
        plan = self.analyzer.generate_improvement_plan([])
        assert plan["steps"] == []
        assert plan["tools"] == []
        assert plan["risks"] == []

    def test_high_priority_adds_risk_entry(self):
        hotspots = [{"file_path": "core/risky.py", "debt_score": 95, "priority": "high", "reasons": []}]
        plan = self.analyzer.generate_improvement_plan(hotspots)
        assert any("core/risky.py" in r for r in plan["risks"])

    def test_high_priority_adds_success_metric(self):
        hotspots = [{"file_path": "core/target.py", "debt_score": 88, "priority": "high", "reasons": []}]
        plan = self.analyzer.generate_improvement_plan(hotspots)
        assert any("core/target.py" in m for m in plan["success_metrics"])


class TestSavePlan:
    def test_save_writes_valid_json(self):
        analyzer = TechnicalDebtAnalyzer(".")
        plan = {"steps": ["step1"], "tools": [], "risks": [], "success_metrics": []}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            analyzer.save_plan(plan, path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["steps"] == ["step1"]
        finally:
            os.unlink(path)

    def test_save_creates_file(self):
        analyzer = TechnicalDebtAnalyzer(".")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "plan.json")
            analyzer.save_plan({"steps": []}, out_path)
            assert os.path.exists(out_path)
