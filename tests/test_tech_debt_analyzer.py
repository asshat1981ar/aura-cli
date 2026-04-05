"""Unit tests for agents/tech_debt_analyzer.py."""
from __future__ import annotations

import unittest

from agents.tech_debt_analyzer import TechnicalDebtAnalyzer


class TestTechnicalDebtAnalyzerInit(unittest.TestCase):
    def test_stores_project_root(self):
        analyzer = TechnicalDebtAnalyzer("/some/project")
        self.assertEqual(analyzer.project_root, "/some/project")

    def test_hotspots_initialised_empty(self):
        analyzer = TechnicalDebtAnalyzer(".")
        self.assertEqual(analyzer.hotspots, [])


class TestAuditCodebase(unittest.TestCase):
    def setUp(self):
        self.analyzer = TechnicalDebtAnalyzer(".")

    def test_returns_list(self):
        result = self.analyzer.audit_codebase()
        self.assertIsInstance(result, list)

    def test_result_non_empty(self):
        result = self.analyzer.audit_codebase()
        self.assertGreater(len(result), 0)

    def test_each_hotspot_has_required_keys(self):
        result = self.analyzer.audit_codebase()
        for hotspot in result:
            self.assertIn("file", hotspot)
            self.assertIn("issue", hotspot)
            self.assertIn("severity", hotspot)
            self.assertIn("recommendation", hotspot)

    def test_hotspots_attribute_updated(self):
        self.analyzer.audit_codebase()
        self.assertGreater(len(self.analyzer.hotspots), 0)

    def test_severity_values_are_valid(self):
        result = self.analyzer.audit_codebase()
        valid_severities = {"high", "medium", "low", "critical"}
        for hotspot in result:
            self.assertIn(hotspot["severity"], valid_severities)


class TestGatherFeedback(unittest.TestCase):
    def setUp(self):
        self.analyzer = TechnicalDebtAnalyzer(".")

    def test_returns_dict(self):
        result = self.analyzer.gather_feedback()
        self.assertIsInstance(result, dict)

    def test_has_developer_feedback_key(self):
        result = self.analyzer.gather_feedback()
        self.assertIn("developer_feedback", result)

    def test_has_qa_feedback_key(self):
        result = self.analyzer.gather_feedback()
        self.assertIn("qa_feedback", result)

    def test_feedback_values_are_lists(self):
        result = self.analyzer.gather_feedback()
        self.assertIsInstance(result["developer_feedback"], list)
        self.assertIsInstance(result["qa_feedback"], list)


class TestProposeImprovements(unittest.TestCase):
    def setUp(self):
        self.analyzer = TechnicalDebtAnalyzer(".")

    def test_returns_list(self):
        result = self.analyzer.propose_improvements()
        self.assertIsInstance(result, list)

    def test_each_improvement_has_required_keys(self):
        result = self.analyzer.propose_improvements()
        for improvement in result:
            self.assertIn("target", improvement)
            self.assertIn("change", improvement)
            self.assertIn("rationale", improvement)

    def test_result_non_empty(self):
        result = self.analyzer.propose_improvements()
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
