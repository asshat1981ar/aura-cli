"""Unit tests for RouterAgent EMA routing logic and ModelAdapter router wiring.

Sprint 6 — s6-model-router-tests
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch, call
import time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brain(stored: dict | None = None):
    """Return a minimal brain mock that supports get/set/remember."""
    brain = MagicMock()
    _store = dict(stored or {})
    brain.get.side_effect = lambda key: _store.get(key)
    brain.set.side_effect = lambda key, val: _store.update({key: val})
    brain.remember.return_value = None
    return brain


def _make_adapter(responses: dict | None = None):
    """Return a ModelAdapter mock with per-model call methods."""
    adapter = MagicMock()
    resp = responses or {}
    for model in ("openai", "gemini", "anthropic", "openrouter", "codex", "copilot", "local"):
        method = f"call_{model}"
        default = resp.get(model, f"response_from_{model}")
        setattr(adapter, method, MagicMock(return_value=default))
    adapter.router = None
    return adapter


# ---------------------------------------------------------------------------
# TestRouterAgentEMA
# ---------------------------------------------------------------------------

class TestRouterAgentEMA(unittest.TestCase):
    """Test EMA score update math in ModelStats.record()."""

    def setUp(self):
        from agents.router import ModelStats
        self.ModelStats = ModelStats

    def test_initial_ema_score(self):
        """New ModelStats starts with ema_score = 0.75 (optimistic prior)."""
        stats = self.ModelStats(name="openai")
        self.assertAlmostEqual(stats.ema_score, 0.75)

    def test_ema_alpha_constant(self):
        """EMA_ALPHA must be 0.2."""
        stats = self.ModelStats(name="openai")
        self.assertAlmostEqual(stats.EMA_ALPHA, 0.2)

    def test_ema_update_on_success(self):
        """After one success: ema = 0.2*1.0 + 0.8*0.75 = 0.80."""
        stats = self.ModelStats(name="openai")
        stats.record(success=True, latency=1.0)
        expected = 0.2 * 1.0 + 0.8 * 0.75
        self.assertAlmostEqual(stats.ema_score, expected, places=6)

    def test_ema_update_on_failure(self):
        """After one failure: ema = 0.2*0.0 + 0.8*0.75 = 0.60."""
        stats = self.ModelStats(name="openai")
        stats.record(success=False, latency=0.5)
        expected = 0.2 * 0.0 + 0.8 * 0.75
        self.assertAlmostEqual(stats.ema_score, expected, places=6)

    def test_ema_multiple_updates(self):
        """EMA converges correctly over several observations."""
        stats = self.ModelStats(name="openai")
        alpha = stats.EMA_ALPHA
        ema = stats.ema_score
        observations = [True, False, True, True, False]
        for success in observations:
            obs = 1.0 if success else 0.0
            ema = alpha * obs + (1 - alpha) * ema
            stats.record(success=success, latency=0.1)
        self.assertAlmostEqual(stats.ema_score, ema, places=6)

    def test_ema_high_failure_rate_decreases_score(self):
        """Repeated failures must drive ema_score below 0.5."""
        stats = self.ModelStats(name="openai")
        for _ in range(20):
            stats.record(success=False, latency=0.1)
        self.assertLess(stats.ema_score, 0.5)

    def test_ema_high_success_rate_increases_score(self):
        """Repeated successes must drive ema_score toward 1.0."""
        stats = self.ModelStats(name="openai")
        for _ in range(30):
            stats.record(success=True, latency=0.1)
        self.assertGreater(stats.ema_score, 0.95)

    def test_latency_accumulates(self):
        """total_latency accumulates across calls regardless of outcome."""
        stats = self.ModelStats(name="gemini")
        stats.record(success=True, latency=2.0)
        stats.record(success=False, latency=3.0)
        self.assertAlmostEqual(stats.total_latency, 5.0, places=6)

    def test_success_failure_counts(self):
        """success_count and failure_count are tracked independently."""
        stats = self.ModelStats(name="anthropic")
        stats.record(success=True, latency=1.0)
        stats.record(success=True, latency=1.0)
        stats.record(success=False, latency=1.0)
        self.assertEqual(stats.success_count, 2)
        self.assertEqual(stats.failure_count, 1)

    def test_avg_latency_calculation(self):
        """avg_latency = total_latency / (success_count + failure_count)."""
        stats = self.ModelStats(name="gemini")
        stats.record(success=True, latency=1.0)
        stats.record(success=True, latency=3.0)
        self.assertAlmostEqual(stats.avg_latency, 2.0, places=6)

    def test_avg_latency_zero_when_no_records(self):
        """avg_latency must be 0 when no observations have been made."""
        stats = self.ModelStats(name="openai")
        self.assertAlmostEqual(stats.avg_latency, 0.0, places=6)

    def test_consecutive_failures_reset_on_success(self):
        """Consecutive-failure counter resets after a success."""
        stats = self.ModelStats(name="openai")
        stats.record(success=False, latency=0.1)
        stats.record(success=False, latency=0.1)
        self.assertEqual(stats.consecutive_failures, 2)
        stats.record(success=True, latency=0.1)
        self.assertEqual(stats.consecutive_failures, 0)


# ---------------------------------------------------------------------------
# TestRouterAgentRouting
# ---------------------------------------------------------------------------

class TestRouterAgentRouting(unittest.TestCase):
    """Test route() returns the highest-EMA model for a given prompt."""

    def _make_router(self, enabled=None, responses=None):
        from agents.router import RouterAgent
        brain = _make_brain()
        adapter = _make_adapter(responses)
        router = RouterAgent(brain, adapter, enabled_models=enabled or ["openai", "gemini"])
        return router, brain, adapter

    def test_route_calls_highest_ema_model(self):
        """route() should call the model with the highest EMA score first."""
        router, _, adapter = self._make_router(enabled=["openai", "gemini"])
        # Manually set gemini higher than openai
        router.stats["gemini"].ema_score = 0.9
        router.stats["openai"].ema_score = 0.6

        result = router.route("hello world prompt")

        adapter.call_gemini.assert_called_once()
        adapter.call_openai.assert_not_called()
        self.assertEqual(result, "response_from_gemini")

    def test_route_falls_back_to_second_model_on_failure(self):
        """If top-ranked model raises, route() falls back to next candidate."""
        router, _, adapter = self._make_router(enabled=["openai", "gemini"])
        router.stats["gemini"].ema_score = 0.9
        router.stats["openai"].ema_score = 0.5
        adapter.call_gemini.side_effect = RuntimeError("gemini down")

        result = router.route("some prompt")

        adapter.call_gemini.assert_called_once()
        adapter.call_openai.assert_called_once()
        self.assertEqual(result, "response_from_openai")

    def test_route_updates_ema_on_success(self):
        """Successful call increments success_count and updates EMA."""
        router, _, adapter = self._make_router(enabled=["openai"])
        before = router.stats["openai"].ema_score
        router.route("a test prompt")
        self.assertEqual(router.stats["openai"].success_count, 1)
        self.assertNotAlmostEqual(router.stats["openai"].ema_score, before)

    def test_route_updates_ema_on_failure(self):
        """Failed call increments failure_count and updates EMA."""
        router, _, adapter = self._make_router(enabled=["openai", "gemini"])
        router.stats["openai"].ema_score = 0.9
        router.stats["gemini"].ema_score = 0.5
        adapter.call_openai.side_effect = RuntimeError("fail")
        router.route("test prompt")
        self.assertEqual(router.stats["openai"].failure_count, 1)

    def test_route_raises_when_all_fail(self):
        """RuntimeError raised when all candidate models fail."""
        router, _, adapter = self._make_router(enabled=["openai", "gemini"])
        adapter.call_openai.side_effect = RuntimeError("openai down")
        adapter.call_gemini.side_effect = RuntimeError("gemini down")

        with self.assertRaises(RuntimeError) as ctx:
            router.route("a prompt that fails everywhere")
        self.assertIn("all candidates exhausted", str(ctx.exception))

    def test_route_skips_cooled_down_models(self):
        """Models in cooldown period are skipped during routing."""
        router, _, adapter = self._make_router(enabled=["openai", "gemini"])
        router.stats["openai"].ema_score = 0.9
        # Put openai in cooldown
        router.stats["openai"].cooldown_until = time.time() + 9999
        router.stats["gemini"].ema_score = 0.5

        result = router.route("test prompt")

        adapter.call_openai.assert_not_called()
        adapter.call_gemini.assert_called_once()
        self.assertEqual(result, "response_from_gemini")

    def test_route_selects_single_available_model(self):
        """Only one non-cooled-down model is available — it must be chosen."""
        router, _, adapter = self._make_router(enabled=["openai", "gemini"])
        router.stats["openai"].cooldown_until = time.time() + 9999
        router.stats["gemini"].ema_score = 0.8

        result = router.route("only one available")
        adapter.call_gemini.assert_called_once()
        self.assertEqual(result, "response_from_gemini")


# ---------------------------------------------------------------------------
# TestRouterAgentFallback
# ---------------------------------------------------------------------------

class TestRouterAgentFallback(unittest.TestCase):
    """Test fallback behaviour when no scores have been recorded."""

    def test_all_models_start_with_equal_ema(self):
        """Fresh router — all models start at ema_score = 0.75."""
        from agents.router import RouterAgent
        brain = _make_brain()
        adapter = _make_adapter()
        router = RouterAgent(brain, adapter, enabled_models=["openai", "gemini", "anthropic"])
        scores = {name: s.ema_score for name, s in router.stats.items()}
        self.assertTrue(all(v == 0.75 for v in scores.values()),
                        f"Expected all 0.75, got {scores}")

    def test_router_initialises_stats_for_all_enabled_models(self):
        """Stats entries are created for every enabled model on init."""
        from agents.router import RouterAgent
        enabled = ["openai", "gemini", "codex"]
        router = RouterAgent(_make_brain(), _make_adapter(), enabled_models=enabled)
        for model in enabled:
            self.assertIn(model, router.stats)

    def test_route_picks_any_model_when_all_equal(self):
        """When all scores are equal, route() still succeeds (picks any)."""
        from agents.router import RouterAgent
        brain = _make_brain()
        adapter = _make_adapter()
        router = RouterAgent(brain, adapter, enabled_models=["openai", "gemini"])
        # All models are at default ema; at least one should succeed
        result = router.route("default fallback prompt")
        self.assertTrue(result.startswith("response_from_"))

    def test_router_loads_persisted_stats_from_brain(self):
        """RouterAgent restores previously persisted stats on construction."""
        from agents.router import RouterAgent, ModelStats
        persisted_stats = {
            "openai": ModelStats(name="openai", ema_score=0.3).to_dict(),
            "gemini": ModelStats(name="gemini", ema_score=0.9).to_dict(),
        }
        brain = _make_brain(stored={"__router_stats__": persisted_stats})
        router = RouterAgent(brain, _make_adapter(), enabled_models=["openai", "gemini"])
        self.assertAlmostEqual(router.stats["openai"].ema_score, 0.3)
        self.assertAlmostEqual(router.stats["gemini"].ema_score, 0.9)


# ---------------------------------------------------------------------------
# TestRouterAgentConfigOverride
# ---------------------------------------------------------------------------

class TestRouterAgentConfigOverride(unittest.TestCase):
    """Test that config-level model_routing overrides are honoured
    by ModelAdapter.respond_for_role().

    The RouterAgent itself routes by prompt; task-type overrides live in
    ModelAdapter.respond_for_role() which reads config['model_routing'].
    """

    def test_respond_for_role_uses_model_routing_config(self):
        """respond_for_role routes to the model specified in model_routing config."""
        from core.model_adapter import ModelAdapter

        adapter = ModelAdapter.__new__(ModelAdapter)
        adapter._mem_cache = {}
        adapter.router = None
        adapter.telemetry_agent = None
        adapter._momento = None

        call_openrouter = MagicMock(return_value="openrouter_response")
        adapter.call_openrouter = call_openrouter
        adapter._get_cached_response = MagicMock(return_value=None)
        adapter._save_to_cache = MagicMock()
        adapter._resolve_local_profile_name = MagicMock(return_value=None)
        adapter._call_with_timeout = MagicMock(side_effect=lambda fn, *args, **kw: fn(*args))

        config_patch = {
            "primary_provider": "openrouter",
            "model_routing": {
                "code_generation": "google/gemini-2.5-pro",
                "planning": "openai/gpt-4o",
            },
        }

        with patch("core.model_adapter.config") as mock_cfg:
            mock_cfg.get.side_effect = lambda key, default=None: config_patch.get(key, default)
            result = adapter.respond_for_role("code_generation", "write me a function")

        call_openrouter.assert_called_once()
        self.assertEqual(result, "openrouter_response")

    def test_respond_for_role_falls_back_when_route_key_absent(self):
        """respond_for_role falls back to respond() when route key not in config."""
        from core.model_adapter import ModelAdapter

        adapter = ModelAdapter.__new__(ModelAdapter)
        adapter._mem_cache = {}
        adapter.router = None
        adapter.telemetry_agent = None
        adapter._momento = None

        fallback_respond = MagicMock(return_value="fallback_response")
        adapter.respond = fallback_respond
        adapter._get_cached_response = MagicMock(return_value=None)
        adapter._save_to_cache = MagicMock()
        adapter._resolve_local_profile_name = MagicMock(return_value=None)
        adapter._call_with_timeout = MagicMock(side_effect=lambda fn, *args, **kw: fn(*args))

        config_patch = {
            "primary_provider": "openrouter",
            "model_routing": {},  # unknown_task not present
        }

        with patch("core.model_adapter.config") as mock_cfg:
            mock_cfg.get.side_effect = lambda key, default=None: config_patch.get(key, default)
            result = adapter.respond_for_role("unknown_task", "some prompt")

        fallback_respond.assert_called_once_with("some prompt")
        self.assertEqual(result, "fallback_response")

    def test_model_routing_task_types_defined_in_default_config(self):
        """Default config contains expected task-type routing keys."""
        from core.config_manager import DEFAULT_CONFIG
        routing = DEFAULT_CONFIG.get("model_routing", {})
        expected_keys = {"code_generation", "planning", "analysis", "critique",
                         "embedding", "fast", "quality"}
        self.assertTrue(expected_keys.issubset(set(routing.keys())),
                        f"Missing routing keys. Found: {set(routing.keys())}")


# ---------------------------------------------------------------------------
# TestModelAdapterRouterWiring
# ---------------------------------------------------------------------------

class TestModelAdapterRouterWiring(unittest.TestCase):
    """Test ModelAdapter.set_router() and that respond() delegates to router."""

    def _make_adapter_instance(self):
        """Build a minimal ModelAdapter instance without real I/O."""
        from core.model_adapter import ModelAdapter
        adapter = ModelAdapter.__new__(ModelAdapter)
        adapter._mem_cache = {}
        adapter.router = None
        adapter.telemetry_agent = None
        adapter._momento = None
        adapter._get_cached_response = MagicMock(return_value=None)
        adapter._save_to_cache = MagicMock()
        return adapter

    def test_set_router_attaches_router(self):
        """set_router() stores the router on adapter.router."""
        adapter = self._make_adapter_instance()
        mock_router = MagicMock()
        adapter.set_router(mock_router)
        self.assertIs(adapter.router, mock_router)

    def test_set_router_replaces_existing_router(self):
        """Calling set_router() twice replaces the old router."""
        adapter = self._make_adapter_instance()
        router_a = MagicMock()
        router_b = MagicMock()
        adapter.set_router(router_a)
        adapter.set_router(router_b)
        self.assertIs(adapter.router, router_b)

    def test_respond_delegates_to_router_when_attached(self):
        """respond() calls router.route(prompt) when a router is attached."""
        adapter = self._make_adapter_instance()

        mock_router = MagicMock()
        mock_router.route.return_value = "routed_response"
        adapter.set_router(mock_router)

        # Patch config and _call_with_timeout
        with patch("core.model_adapter.config") as mock_cfg:
            mock_cfg.get.return_value = "openrouter"
            adapter._call_with_timeout = MagicMock(
                side_effect=lambda fn, *args, **kw: fn(*args)
            )
            result = adapter.respond("test prompt via router")

        mock_router.route.assert_called_once_with("test prompt via router")
        self.assertEqual(result, "routed_response")

    def test_respond_skips_router_when_none(self):
        """respond() does NOT call router when router is None."""
        adapter = self._make_adapter_instance()
        self.assertIsNone(adapter.router)

        fallback_called = []

        def fake_openrouter(prompt):
            fallback_called.append(prompt)
            return "direct_response"

        adapter.call_openrouter = fake_openrouter

        with patch("core.model_adapter.config") as mock_cfg:
            mock_cfg.get.return_value = "openrouter"
            adapter._call_with_timeout = MagicMock(
                side_effect=lambda fn, *args, **kw: fn(*args)
            )
            result = adapter.respond("prompt without router")

        self.assertEqual(fallback_called, ["prompt without router"])
        self.assertEqual(result, "direct_response")

    def test_respond_falls_back_when_router_raises(self):
        """respond() catches router errors and falls back to direct providers."""
        adapter = self._make_adapter_instance()

        mock_router = MagicMock()
        mock_router.route.side_effect = RuntimeError("router failed")
        adapter.set_router(mock_router)

        fallback_called = []

        def fake_openrouter(prompt):
            fallback_called.append(prompt)
            return "fallback_from_openrouter"

        adapter.call_openrouter = fake_openrouter

        with patch("core.model_adapter.config") as mock_cfg:
            mock_cfg.get.return_value = "openrouter"
            adapter._call_with_timeout = MagicMock(
                side_effect=lambda fn, *args, **kw: fn(*args)
            )
            result = adapter.respond("a prompt that breaks the router")

        self.assertEqual(result, "fallback_from_openrouter")

    def test_full_wiring_router_agent_with_adapter(self):
        """End-to-end: RouterAgent wired into ModelAdapter routes correctly."""
        from agents.router import RouterAgent

        brain = _make_brain()
        real_adapter = _make_adapter(responses={"openai": "final_response"})

        router = RouterAgent(brain, real_adapter, enabled_models=["openai"])
        router.stats["openai"].ema_score = 0.9

        from core.model_adapter import ModelAdapter
        adapter = ModelAdapter.__new__(ModelAdapter)
        adapter._mem_cache = {}
        adapter.router = None
        adapter.telemetry_agent = None
        adapter._momento = None
        adapter._get_cached_response = MagicMock(return_value=None)
        adapter._save_to_cache = MagicMock()

        adapter.set_router(router)
        self.assertIs(adapter.router, router)

        # Verify the router itself routes to openai
        result = router.route("end to end prompt")
        real_adapter.call_openai.assert_called_once()
        self.assertEqual(result, "final_response")


if __name__ == "__main__":
    unittest.main()
