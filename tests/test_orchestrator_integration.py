"""Integration tests for LoopOrchestrator — full cycle pipeline with mock agents.

Tests the end-to-end cycle flow, failure routing (re-plan, retry, skip),
circuit breaker behavior, and run_loop stopping conditions.
"""
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.orchestrator import LoopOrchestrator
from core.policy import Policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agents(**overrides):
    """Build a minimal mock agent set.  Override individual agents via kwargs."""
    agents = {}
    for name in ("ingest", "plan", "critique", "synthesize", "act", "verify", "reflect"):
        agent = MagicMock()
        agent.name = name
        agent.run.return_value = {"status": "success"}
        agents[name] = agent
    agents.update(overrides)
    return agents


def _make_orchestrator(agents=None, **kwargs):
    defaults = dict(
        brain=MagicMock(
            get=MagicMock(return_value=None),
            recall_all=MagicMock(return_value=[]),
            recall_recent=MagicMock(return_value=[]),
            recall_with_budget=MagicMock(return_value=[]),
        ),
        project_root=Path("/tmp"),
        policy=Policy.from_config({}),
    )
    defaults.update(kwargs)
    return LoopOrchestrator(agents=agents or _make_agents(), **defaults)


# ---------------------------------------------------------------------------
# Full cycle pipeline
# ---------------------------------------------------------------------------

class TestFullCyclePipeline:
    """Verify that run_cycle invokes the expected phases in order."""

    def test_all_phases_called_on_success(self):
        agents = _make_agents()
        # Act agent must return a change_set for apply to do anything
        agents["act"].run.return_value = {
            "status": "success",
            "change_set": {"changes": []},
        }
        agents["verify"].run.return_value = {"status": "pass", "failures": []}
        orc = _make_orchestrator(agents=agents)
        result = orc.run_cycle("Add a logging utility", dry_run=True)

        assert agents["ingest"].run.called
        assert agents["plan"].run.called
        assert agents["act"].run.called
        assert agents["verify"].run.called
        assert agents["reflect"].run.called
        assert "cycle_id" in result or "stop_reason" in result

    def test_dry_run_does_not_write_files(self, tmp_path):
        agents = _make_agents()
        agents["act"].run.return_value = {
            "status": "success",
            "change_set": {
                "changes": [{
                    "file_path": "new_file.py",
                    "old_code": "",
                    "new_code": "print('hello')",
                    "overwrite_file": True,
                }]
            },
        }
        agents["verify"].run.return_value = {"status": "pass", "failures": []}
        orc = _make_orchestrator(agents=agents, project_root=tmp_path)
        orc.run_cycle("Create new_file.py", dry_run=True)
        assert not (tmp_path / "new_file.py").exists()

    def test_cycle_returns_cycle_summary(self):
        agents = _make_agents()
        agents["verify"].run.return_value = {"status": "pass", "failures": []}
        orc = _make_orchestrator(agents=agents)
        result = orc.run_cycle("Test goal", dry_run=True)
        summary = result.get("cycle_summary", result)
        assert "goal" in summary or "goal" in result
        assert "stop_reason" in summary or "stop_reason" in result


# ---------------------------------------------------------------------------
# Failure routing in the pipeline
# ---------------------------------------------------------------------------

class TestFailureRouting:
    """Verify that verification failures trigger the right recovery path."""

    def test_verify_fail_triggers_act_retry(self):
        agents = _make_agents()
        call_count = {"n": 0}

        def act_side_effect(data):
            call_count["n"] += 1
            return {"status": "success", "change_set": {"changes": []}}

        agents["act"].run.side_effect = act_side_effect
        # First verify fails, second passes
        agents["verify"].run.side_effect = [
            {"status": "fail", "failures": ["assertion error in test_foo"]},
            {"status": "pass", "failures": []},
        ]
        orc = _make_orchestrator(agents=agents)
        orc.run_cycle("Fix bug", dry_run=True)
        # Act should be called at least twice (initial + retry)
        assert call_count["n"] >= 2

    def test_structural_failure_triggers_replan(self):
        agents = _make_agents()
        plan_count = {"n": 0}

        def plan_side_effect(data):
            plan_count["n"] += 1
            return {"status": "success", "steps": []}

        agents["plan"].run.side_effect = plan_side_effect
        agents["act"].run.return_value = {"status": "success", "change_set": {"changes": []}}
        agents["verify"].run.side_effect = [
            {"status": "fail", "failures": ["architecture mismatch detected"]},
            {"status": "pass", "failures": []},
        ]
        orc = _make_orchestrator(agents=agents)
        orc.run_cycle("Refactor module", dry_run=True)
        # Plan should be called at least twice (initial + re-plan)
        assert plan_count["n"] >= 2

    def test_external_failure_triggers_skip(self):
        agents = _make_agents()
        agents["act"].run.return_value = {"status": "success", "change_set": {"changes": []}}
        agents["verify"].run.return_value = {
            "status": "fail",
            "failures": ["ModuleNotFoundError: No module named 'missing_dep'"],
        }
        orc = _make_orchestrator(agents=agents)
        result = orc.run_cycle("Install dependency", dry_run=True)
        summary = result.get("cycle_summary", result)
        # External failures should not cause infinite retries
        stop = summary.get("stop_reason") or result.get("stop_reason", "")
        assert stop != ""  # Should have a stop reason


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """Verify the circuit breaker prevents runaway cycles."""

    def test_circuit_breaker_opens_after_threshold(self):
        orc = _make_orchestrator()
        # Trip the breaker by recording failures
        for _ in range(6):
            orc._circuit_breaker.record_failure()

        assert orc._circuit_breaker.is_open()
        result = orc.run_cycle("Should be blocked", dry_run=True)
        summary = result.get("cycle_summary", result)
        stop = summary.get("stop_reason") or result.get("stop_reason")
        assert stop == "CIRCUIT_BREAKER_OPEN"

    def test_circuit_breaker_resets_after_cooldown(self):
        orc = _make_orchestrator()
        orc._circuit_breaker._threshold = 2
        orc._circuit_breaker._cooldown_s = 0.1
        orc._circuit_breaker.record_failure()
        orc._circuit_breaker.record_failure()
        orc._circuit_breaker.record_failure()
        assert orc._circuit_breaker.is_open()
        time.sleep(0.15)
        assert not orc._circuit_breaker.is_open()


# ---------------------------------------------------------------------------
# run_loop stopping conditions
# ---------------------------------------------------------------------------

class TestRunLoop:
    """Verify run_loop respects max_cycles and policy stopping."""

    def test_run_loop_respects_max_cycles(self):
        agents = _make_agents()
        agents["verify"].run.return_value = {"status": "pass", "failures": []}
        orc = _make_orchestrator(agents=agents)
        result = orc.run_loop("Implement feature", max_cycles=2, dry_run=True)
        assert "stop_reason" in result
        history = result.get("history", [])
        assert len(history) <= 2

    def test_run_loop_stops_on_verification_pass(self):
        agents = _make_agents()
        agents["verify"].run.return_value = {"status": "pass", "failures": []}
        orc = _make_orchestrator(agents=agents)
        result = orc.run_loop("Simple fix", max_cycles=5, dry_run=True)
        history = result.get("history", [])
        # Should stop after first successful cycle, not run all 5
        assert len(history) <= 2


# ---------------------------------------------------------------------------
# LLMCache integration (B2 verification)
# ---------------------------------------------------------------------------

class TestLLMCacheIntegration:
    """Verify the extracted LLMCache works correctly in isolation."""

    def test_l0_get_put(self):
        from core.llm_cache import LLMCache
        cache = LLMCache(ttl_seconds=60)
        assert cache.get("hello") is None
        cache.put("hello", "world")
        assert cache.get("hello") == "world"

    def test_l0_lru_eviction(self):
        from core.llm_cache import LLMCache
        cache = LLMCache(ttl_seconds=60, max_l0_entries=3)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.put("c", "3")
        cache.put("d", "4")  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == "2"

    def test_l2_sqlite_persist(self, tmp_path):
        import sqlite3
        from core.llm_cache import LLMCache
        db = sqlite3.connect(str(tmp_path / "test.db"))
        cache = LLMCache(ttl_seconds=3600)
        cache.enable(db)
        cache.put("prompt1", "response1")
        # Create a fresh cache instance to verify persistence
        cache2 = LLMCache(ttl_seconds=3600)
        cache2.enable(db)
        assert cache2.get("prompt1") == "response1"

    def test_preload_populates_l0(self, tmp_path):
        import sqlite3
        from core.llm_cache import LLMCache
        db = sqlite3.connect(str(tmp_path / "test.db"))
        cache = LLMCache(ttl_seconds=3600)
        cache.enable(db)
        cache.put("p1", "r1")
        cache.put("p2", "r2")
        # New cache with same DB
        cache2 = LLMCache(ttl_seconds=3600)
        cache2.enable(db)  # enable calls preload
        # L0 should have the entries
        assert cache2.get("p1") == "r1"
        assert cache2.get("p2") == "r2"

    def test_model_adapter_delegates_to_cache(self):
        """ModelAdapter.enable_cache/respond should use LLMCache."""
        from core.model_adapter import ModelAdapter
        adapter = ModelAdapter()
        # Pre-populate cache
        adapter._cache.put("test prompt", "cached response")
        assert adapter._get_cached_response("test prompt") == "cached response"
