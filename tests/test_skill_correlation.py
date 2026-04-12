"""Tests for core.skill_correlation — skill correlation matrix."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from core.skill_correlation import SkillCorrelationMatrix, SkillOutcome


@pytest.fixture
def tmp_store(tmp_path):
    """Return a temporary path for the correlation store."""
    return tmp_path / "skill_correlations.json"


@pytest.fixture
def matrix(tmp_store):
    """Return a fresh SkillCorrelationMatrix with a temp store."""
    return SkillCorrelationMatrix(store_path=tmp_store)


# ── Record cycle and verify matrix updates ────────────────────────────────


class TestRecordCycle:
    def test_records_pairwise_correlations(self, matrix):
        outcomes = [
            SkillOutcome(skill_name="linter", goal_type="bug_fix", success=True),
            SkillOutcome(skill_name="type_checker", goal_type="bug_fix", success=True),
        ]
        matrix.record_cycle(outcomes, cycle_success=True)

        pair = matrix.matrix["linter"]["type_checker"]
        assert pair["total"] == 1
        assert pair["co_success"] == 1
        assert pair["co_failure"] == 0

    def test_records_failure(self, matrix):
        outcomes = [
            SkillOutcome(skill_name="linter", goal_type="bug_fix", success=False),
            SkillOutcome(skill_name="type_checker", goal_type="bug_fix", success=True),
        ]
        matrix.record_cycle(outcomes, cycle_success=False)

        pair = matrix.matrix["linter"]["type_checker"]
        assert pair["total"] == 1
        assert pair["co_success"] == 0
        assert pair["co_failure"] == 1

    def test_mirrors_are_symmetric(self, matrix):
        outcomes = [
            SkillOutcome(skill_name="a", goal_type="feature", success=True),
            SkillOutcome(skill_name="b", goal_type="feature", success=True),
        ]
        matrix.record_cycle(outcomes, cycle_success=True)

        assert matrix.matrix["a"]["b"]["total"] == matrix.matrix["b"]["a"]["total"]
        assert matrix.matrix["a"]["b"]["co_success"] == matrix.matrix["b"]["a"]["co_success"]

    def test_updates_skill_rates(self, matrix):
        outcomes = [
            SkillOutcome(skill_name="linter", goal_type="bug_fix", success=True),
            SkillOutcome(skill_name="scanner", goal_type="bug_fix", success=False),
        ]
        matrix.record_cycle(outcomes, cycle_success=True)

        assert matrix.skill_rates["bug_fix"]["linter"]["success"] == 1
        assert matrix.skill_rates["bug_fix"]["linter"]["total"] == 1
        assert matrix.skill_rates["bug_fix"]["scanner"]["success"] == 0
        assert matrix.skill_rates["bug_fix"]["scanner"]["total"] == 1

    def test_multiple_cycles_accumulate(self, matrix):
        outcomes = [
            SkillOutcome(skill_name="a", goal_type="feature", success=True),
            SkillOutcome(skill_name="b", goal_type="feature", success=True),
        ]
        matrix.record_cycle(outcomes, cycle_success=True)
        matrix.record_cycle(outcomes, cycle_success=True)

        assert matrix.matrix["a"]["b"]["total"] == 2
        assert matrix.matrix["a"]["b"]["co_success"] == 2


# ── Correlation calculation ───────────────────────────────────────────────


class TestGetCorrelation:
    def test_positive_correlation(self, matrix):
        outcomes = [
            SkillOutcome(skill_name="a", goal_type="feature", success=True),
            SkillOutcome(skill_name="b", goal_type="feature", success=True),
        ]
        matrix.record_cycle(outcomes, cycle_success=True)
        assert matrix.get_correlation("a", "b") == 1.0

    def test_negative_correlation(self, matrix):
        outcomes = [
            SkillOutcome(skill_name="a", goal_type="feature", success=False),
            SkillOutcome(skill_name="b", goal_type="feature", success=False),
        ]
        matrix.record_cycle(outcomes, cycle_success=False)
        assert matrix.get_correlation("a", "b") == -1.0

    def test_zero_correlation_unknown_pair(self, matrix):
        assert matrix.get_correlation("x", "y") == 0.0

    def test_mixed_correlation(self, matrix):
        outcomes = [
            SkillOutcome(skill_name="a", goal_type="feature", success=True),
            SkillOutcome(skill_name="b", goal_type="feature", success=True),
        ]
        # One success cycle, one failure cycle
        matrix.record_cycle(outcomes, cycle_success=True)
        matrix.record_cycle(outcomes, cycle_success=False)
        # co_success=1, co_failure=1, total=2 => (1-1)/2 = 0.0
        assert matrix.get_correlation("a", "b") == 0.0


# ── Suggest skills ────────────────────────────────────────────────────────


class TestSuggestSkills:
    def test_suggests_correlated_skills(self, matrix):
        # Build correlation: a+b succeed together, a+c succeed together
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="a", goal_type="feature", success=True),
                SkillOutcome(skill_name="b", goal_type="feature", success=True),
                SkillOutcome(skill_name="c", goal_type="feature", success=True),
            ],
            cycle_success=True,
        )

        suggestions = matrix.suggest_skills(["a"], top_k=3)
        names = [s[0] for s in suggestions]
        assert "b" in names
        assert "c" in names

    def test_does_not_suggest_base_skills(self, matrix):
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="a", goal_type="feature", success=True),
                SkillOutcome(skill_name="b", goal_type="feature", success=True),
            ],
            cycle_success=True,
        )

        suggestions = matrix.suggest_skills(["a", "b"], top_k=3)
        names = [s[0] for s in suggestions]
        assert "a" not in names
        assert "b" not in names

    def test_empty_suggestions_for_unknown_skills(self, matrix):
        suggestions = matrix.suggest_skills(["unknown_skill"], top_k=3)
        assert suggestions == []

    def test_excludes_negatively_correlated(self, matrix):
        # Only failure cycles => negative correlation
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="a", goal_type="feature", success=False),
                SkillOutcome(skill_name="bad", goal_type="feature", success=False),
            ],
            cycle_success=False,
        )

        suggestions = matrix.suggest_skills(["a"], top_k=3)
        names = [s[0] for s in suggestions]
        assert "bad" not in names


# ── Discover clusters ─────────────────────────────────────────────────────


class TestDiscoverClusters:
    def test_discovers_cluster(self, matrix):
        # Three skills that always succeed together
        for _ in range(3):
            matrix.record_cycle(
                [
                    SkillOutcome(skill_name="x", goal_type="feature", success=True),
                    SkillOutcome(skill_name="y", goal_type="feature", success=True),
                    SkillOutcome(skill_name="z", goal_type="feature", success=True),
                ],
                cycle_success=True,
            )

        clusters = matrix.discover_clusters(min_correlation=0.5, min_size=2)
        assert len(clusters) >= 1
        # All three should be in one cluster
        cluster_skills = clusters[0]
        assert "x" in cluster_skills
        assert "y" in cluster_skills
        assert "z" in cluster_skills

    def test_no_clusters_below_threshold(self, matrix):
        # Only failure cycles => negative correlation, no clusters
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="a", goal_type="feature", success=False),
                SkillOutcome(skill_name="b", goal_type="feature", success=False),
            ],
            cycle_success=False,
        )

        clusters = matrix.discover_clusters(min_correlation=0.5, min_size=2)
        assert clusters == []

    def test_min_size_filter(self, matrix):
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="solo", goal_type="feature", success=True),
            ],
            cycle_success=True,
        )

        clusters = matrix.discover_clusters(min_correlation=0.5, min_size=2)
        assert clusters == []


# ── Success rate tracking ─────────────────────────────────────────────────


class TestSuccessRate:
    def test_rate_by_goal_type(self, matrix):
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="linter", goal_type="bug_fix", success=True),
            ],
            cycle_success=True,
        )
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="linter", goal_type="bug_fix", success=False),
            ],
            cycle_success=False,
        )

        rate = matrix.get_skill_success_rate("linter", goal_type="bug_fix")
        assert rate == 0.5

    def test_rate_aggregated(self, matrix):
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="linter", goal_type="bug_fix", success=True),
            ],
            cycle_success=True,
        )
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="linter", goal_type="feature", success=True),
            ],
            cycle_success=True,
        )
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="linter", goal_type="refactor", success=False),
            ],
            cycle_success=False,
        )

        rate = matrix.get_skill_success_rate("linter")
        # 2 successes out of 3 total
        assert abs(rate - 2 / 3) < 0.01

    def test_rate_unknown_skill(self, matrix):
        rate = matrix.get_skill_success_rate("nonexistent")
        assert rate == 0.0

    def test_rate_unknown_goal_type(self, matrix):
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="linter", goal_type="bug_fix", success=True),
            ],
            cycle_success=True,
        )
        # Ask for a goal_type that has no data — falls through to aggregated
        rate = matrix.get_skill_success_rate("linter", goal_type="unknown_type")
        # goal_type not in skill_rates, so aggregated path: 1/1 = 1.0
        assert rate == 1.0


# ── Summary generation ────────────────────────────────────────────────────


class TestSummary:
    def test_summary_structure(self, matrix):
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="a", goal_type="feature", success=True),
                SkillOutcome(skill_name="b", goal_type="feature", success=True),
            ],
            cycle_success=True,
        )

        summary = matrix.get_summary()
        assert "total_skills_tracked" in summary
        assert "total_pairs" in summary
        assert "top_correlations" in summary
        assert "clusters" in summary
        assert summary["total_skills_tracked"] == 2
        assert summary["total_pairs"] == 1

    def test_empty_summary(self, matrix):
        summary = matrix.get_summary()
        assert summary["total_skills_tracked"] == 0
        assert summary["total_pairs"] == 0
        assert summary["top_correlations"] == []
        assert summary["clusters"] == []

    def test_summary_filters_low_correlation(self, matrix):
        # Mixed results that yield very low correlation
        outcomes = [
            SkillOutcome(skill_name="a", goal_type="feature", success=True),
            SkillOutcome(skill_name="b", goal_type="feature", success=True),
        ]
        # 10 success cycles + 10 failure cycles => correlation = 0.0
        for _ in range(10):
            matrix.record_cycle(outcomes, cycle_success=True)
        for _ in range(10):
            matrix.record_cycle(outcomes, cycle_success=False)

        summary = matrix.get_summary()
        # Correlation is 0.0, which is not > 0.1, so no top correlations
        assert summary["top_correlations"] == []


# ── Persistence ───────────────────────────────────────────────────────────


class TestPersistence:
    def test_save_and_reload(self, tmp_store):
        m1 = SkillCorrelationMatrix(store_path=tmp_store)
        m1.record_cycle(
            [
                SkillOutcome(skill_name="a", goal_type="feature", success=True),
                SkillOutcome(skill_name="b", goal_type="feature", success=True),
            ],
            cycle_success=True,
        )

        # Create a new instance that loads from the same file
        m2 = SkillCorrelationMatrix(store_path=tmp_store)

        assert m2.matrix["a"]["b"]["co_success"] == 1
        assert m2.matrix["a"]["b"]["total"] == 1
        assert m2.get_correlation("a", "b") == 1.0

    def test_save_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "nested" / "dir" / "correlations.json"
        m = SkillCorrelationMatrix(store_path=deep_path)
        m.record_cycle(
            [
                SkillOutcome(skill_name="x", goal_type="feature", success=True),
            ],
            cycle_success=True,
        )
        assert deep_path.exists()

    def test_reload_preserves_skill_rates(self, tmp_store):
        m1 = SkillCorrelationMatrix(store_path=tmp_store)
        m1.record_cycle(
            [
                SkillOutcome(skill_name="linter", goal_type="bug_fix", success=True),
            ],
            cycle_success=True,
        )

        m2 = SkillCorrelationMatrix(store_path=tmp_store)
        assert m2.skill_rates["bug_fix"]["linter"]["success"] == 1

    def test_reload_preserves_clusters(self, tmp_store):
        m1 = SkillCorrelationMatrix(store_path=tmp_store)
        for _ in range(3):
            m1.record_cycle(
                [
                    SkillOutcome(skill_name="a", goal_type="feature", success=True),
                    SkillOutcome(skill_name="b", goal_type="feature", success=True),
                ],
                cycle_success=True,
            )
        m1.discover_clusters(min_correlation=0.5, min_size=2)
        m1._save()

        m2 = SkillCorrelationMatrix(store_path=tmp_store)
        assert len(m2.clusters) >= 1

    def test_handles_corrupt_file(self, tmp_store):
        tmp_store.write_text("not json at all {{{")
        m = SkillCorrelationMatrix(store_path=tmp_store)
        # Should not raise, just start empty
        assert m.get_correlation("a", "b") == 0.0


# ── Empty matrix edge cases ──────────────────────────────────────────────


class TestEdgeCases:
    def test_record_single_skill_cycle(self, matrix):
        # Single skill => no pairs to record
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="solo", goal_type="feature", success=True),
            ],
            cycle_success=True,
        )
        assert matrix.skill_rates["feature"]["solo"]["success"] == 1
        # No pairwise entries
        assert len(matrix.matrix) == 0

    def test_record_empty_outcomes(self, matrix):
        # Empty list => nothing recorded, no crash
        matrix.record_cycle([], cycle_success=True)
        assert len(matrix.matrix) == 0

    def test_suggest_from_empty_matrix(self, matrix):
        suggestions = matrix.suggest_skills(["anything"])
        assert suggestions == []

    def test_discover_clusters_empty_matrix(self, matrix):
        clusters = matrix.discover_clusters()
        assert clusters == []

    def test_correlation_is_symmetric(self, matrix):
        matrix.record_cycle(
            [
                SkillOutcome(skill_name="a", goal_type="feature", success=True),
                SkillOutcome(skill_name="b", goal_type="feature", success=True),
            ],
            cycle_success=True,
        )
        assert matrix.get_correlation("a", "b") == matrix.get_correlation("b", "a")
