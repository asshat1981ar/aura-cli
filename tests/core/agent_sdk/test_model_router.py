"""Tests for core/agent_sdk/model_router.py — AdaptiveModelRouter."""

import json
import pytest
from pathlib import Path

from core.agent_sdk.model_router import (
    AdaptiveModelRouter,
    MODEL_TIERS,
    MODEL_TO_TIER,
    TIER_TO_MODEL,
    TIER_ORDER,
)


def _router(tmp_path, **kwargs) -> AdaptiveModelRouter:
    return AdaptiveModelRouter(
        stats_path=tmp_path / "stats.json",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_three_tiers(self):
        assert len(MODEL_TIERS) == 3

    def test_tier_names(self):
        tiers = {t["tier"] for t in MODEL_TIERS}
        assert tiers == {"fast", "standard", "powerful"}

    def test_tier_order_cheapest_first(self):
        assert TIER_ORDER[0] == "fast"
        assert TIER_ORDER[-1] == "powerful"

    def test_model_to_tier_round_trips(self):
        for tier, model in TIER_TO_MODEL.items():
            assert MODEL_TO_TIER[model] == tier


# ---------------------------------------------------------------------------
# Init / load
# ---------------------------------------------------------------------------

class TestInit:
    def test_empty_stats_when_file_missing(self, tmp_path):
        r = _router(tmp_path)
        assert r.get_stats() == {}

    def test_loads_existing_stats(self, tmp_path):
        p = tmp_path / "stats.json"
        data = {"bug_fix": {"standard": {"attempts": 5, "successes": 4,
                "consecutive_failures": 0, "consecutive_successes": 4, "ema_score": 0.8}}}
        p.write_text(json.dumps(data))
        r = AdaptiveModelRouter(stats_path=p)
        stats = r.get_stats()
        assert "bug_fix" in stats

    def test_corrupted_file_starts_empty(self, tmp_path):
        p = tmp_path / "stats.json"
        p.write_text("{BROKEN")
        r = AdaptiveModelRouter(stats_path=p)
        assert r.get_stats() == {}


# ---------------------------------------------------------------------------
# record_outcome
# ---------------------------------------------------------------------------

class TestRecordOutcome:
    def test_record_creates_entry(self, tmp_path):
        r = _router(tmp_path)
        r.record_outcome("bug_fix", "claude-sonnet-4-6", True)
        stats = r.get_stats()
        assert "bug_fix" in stats
        assert "standard" in stats["bug_fix"]
        assert stats["bug_fix"]["standard"]["attempts"] == 1

    def test_success_increments_successes(self, tmp_path):
        r = _router(tmp_path)
        r.record_outcome("t", "claude-sonnet-4-6", True)
        assert r.get_stats()["t"]["standard"]["successes"] == 1

    def test_success_increments_consecutive_successes(self, tmp_path):
        r = _router(tmp_path)
        r.record_outcome("t", "claude-sonnet-4-6", True)
        r.record_outcome("t", "claude-sonnet-4-6", True)
        assert r.get_stats()["t"]["standard"]["consecutive_successes"] == 2

    def test_failure_increments_consecutive_failures(self, tmp_path):
        r = _router(tmp_path)
        r.record_outcome("t", "claude-sonnet-4-6", False)
        assert r.get_stats()["t"]["standard"]["consecutive_failures"] == 1

    def test_success_resets_consecutive_failures(self, tmp_path):
        r = _router(tmp_path)
        r.record_outcome("t", "claude-sonnet-4-6", False)
        r.record_outcome("t", "claude-sonnet-4-6", True)
        assert r.get_stats()["t"]["standard"]["consecutive_failures"] == 0

    def test_failure_resets_consecutive_successes(self, tmp_path):
        r = _router(tmp_path)
        r.record_outcome("t", "claude-sonnet-4-6", True)
        r.record_outcome("t", "claude-sonnet-4-6", False)
        assert r.get_stats()["t"]["standard"]["consecutive_successes"] == 0

    def test_ema_updates_toward_success(self, tmp_path):
        r = _router(tmp_path, ema_alpha=0.5)
        r.record_outcome("t", "claude-sonnet-4-6", True)
        ema = r.get_stats()["t"]["standard"]["ema_score"]
        assert ema > 0.5

    def test_ema_updates_toward_failure(self, tmp_path):
        r = _router(tmp_path, ema_alpha=0.5)
        r.record_outcome("t", "claude-sonnet-4-6", False)
        ema = r.get_stats()["t"]["standard"]["ema_score"]
        assert ema < 0.5

    def test_stats_persisted_to_file(self, tmp_path):
        p = tmp_path / "stats.json"
        r = AdaptiveModelRouter(stats_path=p)
        r.record_outcome("t", "claude-haiku-4-5", True)
        data = json.loads(p.read_text())
        assert "t" in data


# ---------------------------------------------------------------------------
# select_model
# ---------------------------------------------------------------------------

class TestSelectModel:
    def test_no_stats_returns_standard(self, tmp_path):
        r = _router(tmp_path)
        model = r.select_model("unknown_type")
        assert model == TIER_TO_MODEL["standard"]

    def test_well_performing_standard_selected(self, tmp_path):
        r = _router(tmp_path, min_success_rate=0.7, escalation_threshold=2)
        for _ in range(5):
            r.record_outcome("t", "claude-sonnet-4-6", True)
        model = r.select_model("t")
        assert model == TIER_TO_MODEL["standard"]

    def test_consecutive_failures_escalates(self, tmp_path):
        r = _router(tmp_path, escalation_threshold=2)
        # Record enough successes to establish standard, then fail
        for _ in range(3):
            r.record_outcome("t", "claude-sonnet-4-6", True)
        for _ in range(3):
            r.record_outcome("t", "claude-sonnet-4-6", False)
        model = r.select_model("t")
        # Should escalate past standard
        assert model != TIER_TO_MODEL["standard"]

    def test_fallback_powerful_when_all_fail(self, tmp_path):
        r = _router(tmp_path, escalation_threshold=1, min_success_rate=0.9)
        # Record failures on all tiers
        for _ in range(3):
            r.record_outcome("t", "claude-haiku-4-5", False)
            r.record_outcome("t", "claude-sonnet-4-6", False)
        model = r.select_model("t")
        assert model == TIER_TO_MODEL["powerful"]


# ---------------------------------------------------------------------------
# escalate
# ---------------------------------------------------------------------------

class TestEscalate:
    def test_fast_escalates_to_standard(self, tmp_path):
        r = _router(tmp_path)
        result = r.escalate("t", TIER_TO_MODEL["fast"])
        assert result == TIER_TO_MODEL["standard"]

    def test_standard_escalates_to_powerful(self, tmp_path):
        r = _router(tmp_path)
        result = r.escalate("t", TIER_TO_MODEL["standard"])
        assert result == TIER_TO_MODEL["powerful"]

    def test_powerful_stays_at_powerful(self, tmp_path):
        r = _router(tmp_path)
        result = r.escalate("t", TIER_TO_MODEL["powerful"])
        assert result == TIER_TO_MODEL["powerful"]

    def test_unknown_model_treated_as_standard(self, tmp_path):
        r = _router(tmp_path)
        result = r.escalate("t", "unknown_model_xyz")
        assert result == TIER_TO_MODEL["powerful"]
