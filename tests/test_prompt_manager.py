"""
Unit tests for agents/prompt_manager.py.

Covers:
- PromptCacheEntry dataclass behaviour
- PromptCache LRU + TTL logic
- SYSTEM_PROMPTS dict and get_system_prompt()
- render_prompt() template rendering and caching
- Legacy alias assertions
- Global cache helpers
"""

import time
import unittest

from agents.prompt_manager import (
    PromptCacheEntry,
    PromptCache,
    SYSTEM_PROMPTS,
    PLANNER_PROMPT_TEMPLATE,
    CRITIC_PROMPT_TEMPLATE,
    CODER_PROMPT_TEMPLATE,
    PLANNER_COT_PROMPT_TEMPLATE,
    CRITIC_COT_PROMPT_TEMPLATE,
    CODER_COT_PROMPT_TEMPLATE,
    get_system_prompt,
    render_prompt,
    get_cached_prompt_stats,
    clear_prompt_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLANNER_PARAMS = {
    "goal": "Refactor the goal queue",
    "memory": "some memory",
    "similar": "similar past problems",
    "weakness": "none",
    "backfill_instr": "",
}

_CRITIC_PARAMS = {
    "target_type": "plan",
    "task": "Evaluate plan quality",
    "target_content": "step 1: do x",
    "memory": "context here",
}

_CODER_PARAMS = {
    "task": "Implement feature Y",
    "memory": "related context",
    "code_section": "",
    "tests_section": "",
    "feedback_section": "",
}


# ===========================================================================
# TestPromptCacheEntry
# ===========================================================================


class TestPromptCacheEntry(unittest.TestCase):
    """Tests for the PromptCacheEntry dataclass."""

    def test_touch_increments_count(self):
        """touch() should increment access_count from 0 to 1."""
        entry = PromptCacheEntry(prompt="hello")
        self.assertEqual(entry.access_count, 0)
        entry.touch()
        self.assertEqual(entry.access_count, 1)

    def test_touch_increments_count_multiple_times(self):
        """Multiple touch() calls should accumulate the count."""
        entry = PromptCacheEntry(prompt="hello")
        for i in range(5):
            entry.touch()
        self.assertEqual(entry.access_count, 5)

    def test_touch_updates_last_accessed(self):
        """touch() should update last_accessed to a time >= created_at."""
        entry = PromptCacheEntry(prompt="hello")
        before = entry.last_accessed
        # Small sleep to guarantee time progresses.
        time.sleep(0.01)
        entry.touch()
        self.assertGreaterEqual(entry.last_accessed, before)


# ===========================================================================
# TestPromptCache
# ===========================================================================


class TestPromptCache(unittest.TestCase):
    """Tests for PromptCache LRU + TTL behaviour."""

    def setUp(self):
        self.cache = PromptCache(max_size=10, ttl_seconds=3600)

    # ------------------------------------------------------------------
    # Basic get / set
    # ------------------------------------------------------------------

    def test_set_and_get(self):
        """set() followed by get() should return the stored prompt."""
        params = {"key": "value"}
        self.cache.set("planner", params, "my prompt text")
        result = self.cache.get("planner", params)
        self.assertEqual(result, "my prompt text")

    def test_get_miss_returns_none(self):
        """get() for a key that was never set should return None."""
        result = self.cache.get("planner", {"nonexistent": "key"})
        self.assertIsNone(result)

    def test_different_params_are_different_entries(self):
        """Two sets with different params should not collide."""
        self.cache.set("planner", {"a": "1"}, "prompt-a")
        self.cache.set("planner", {"a": "2"}, "prompt-b")
        self.assertEqual(self.cache.get("planner", {"a": "1"}), "prompt-a")
        self.assertEqual(self.cache.get("planner", {"a": "2"}), "prompt-b")

    def test_different_template_names_are_different_entries(self):
        """Same params under different template names must not alias."""
        params = {"x": "1"}
        self.cache.set("planner", params, "planner-prompt")
        self.cache.set("coder", params, "coder-prompt")
        self.assertEqual(self.cache.get("planner", params), "planner-prompt")
        self.assertEqual(self.cache.get("coder", params), "coder-prompt")

    # ------------------------------------------------------------------
    # TTL expiry
    # ------------------------------------------------------------------

    def test_ttl_expiry(self):
        """Entries older than ttl_seconds should be evicted on get()."""
        short_cache = PromptCache(max_size=10, ttl_seconds=0.01)
        short_cache.set("planner", {"g": "goal"}, "will expire")
        time.sleep(0.02)
        result = short_cache.get("planner", {"g": "goal"})
        self.assertIsNone(result)

    def test_entry_within_ttl_is_returned(self):
        """Entries younger than ttl_seconds should still be returned."""
        short_cache = PromptCache(max_size=10, ttl_seconds=60)
        short_cache.set("planner", {"g": "goal"}, "still valid")
        result = short_cache.get("planner", {"g": "goal"})
        self.assertEqual(result, "still valid")

    # ------------------------------------------------------------------
    # Eviction at capacity
    # ------------------------------------------------------------------

    def test_eviction_at_capacity(self):
        """Adding a third item to a max_size=2 cache evicts the oldest."""
        small_cache = PromptCache(max_size=2, ttl_seconds=3600)
        small_cache.set("t", {"k": "1"}, "first")
        time.sleep(0.01)  # ensure last_accessed ordering
        small_cache.set("t", {"k": "2"}, "second")
        time.sleep(0.01)
        # Adding a third item should evict "first" (oldest last_accessed).
        small_cache.set("t", {"k": "3"}, "third")

        self.assertIsNone(small_cache.get("t", {"k": "1"}))
        self.assertEqual(small_cache.get("t", {"k": "2"}), "second")
        self.assertEqual(small_cache.get("t", {"k": "3"}), "third")

    def test_size_never_exceeds_max_size(self):
        """Cache size must never exceed max_size after multiple inserts."""
        small_cache = PromptCache(max_size=3, ttl_seconds=3600)
        for i in range(10):
            small_cache.set("t", {"i": str(i)}, f"prompt-{i}")
        stats = small_cache.get_stats()
        self.assertLessEqual(stats["size"], 3)

    # ------------------------------------------------------------------
    # Stats tracking
    # ------------------------------------------------------------------

    def test_stats_initial_state(self):
        """A fresh cache should report zero hits, misses and hit_rate=0."""
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["hit_rate"], 0)
        self.assertEqual(stats["size"], 0)
        self.assertEqual(stats["max_size"], 10)

    def test_stats_tracking(self):
        """hits and misses should be counted correctly after set + get."""
        params = {"g": "goal"}
        self.cache.set("planner", params, "some prompt")

        # First get: hit
        self.cache.get("planner", params)
        # Second get on a missing key: miss
        self.cache.get("planner", {"other": "params"})

        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5, places=5)

    def test_stats_hit_rate_all_hits(self):
        """hit_rate should be 1.0 when every get is a cache hit."""
        params = {"g": "goal"}
        self.cache.set("planner", params, "prompt")
        self.cache.get("planner", params)
        self.cache.get("planner", params)

        stats = self.cache.get_stats()
        self.assertEqual(stats["hit_rate"], 1.0)

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def test_clear_resets_everything(self):
        """clear() should empty the cache and reset hit/miss counters."""
        self.cache.set("planner", {"g": "goal"}, "prompt")
        self.cache.get("planner", {"g": "goal"})  # hit
        self.cache.get("planner", {"missing": "k"})  # miss

        self.cache.clear()

        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["size"], 0)

    def test_clear_prevents_stale_retrieval(self):
        """After clear(), previously cached prompts must not be returned."""
        params = {"g": "goal"}
        self.cache.set("planner", params, "old prompt")
        self.cache.clear()
        self.assertIsNone(self.cache.get("planner", params))


# ===========================================================================
# TestSystemPrompts
# ===========================================================================


class TestSystemPrompts(unittest.TestCase):
    """Tests for SYSTEM_PROMPTS and get_system_prompt()."""

    def test_all_roles_have_prompts(self):
        """All expected roles must be present in SYSTEM_PROMPTS."""
        expected_roles = {"planner", "critic", "coder", "synthesizer", "reflector"}
        self.assertEqual(expected_roles, set(SYSTEM_PROMPTS.keys()))

    def test_all_prompts_are_non_empty_strings(self):
        """Every system prompt must be a non-empty string."""
        for role, prompt in SYSTEM_PROMPTS.items():
            with self.subTest(role=role):
                self.assertIsInstance(prompt, str)
                self.assertTrue(len(prompt) > 0)

    def test_get_system_prompt_known_role(self):
        """get_system_prompt('planner') should return the planner prompt."""
        result = get_system_prompt("planner")
        self.assertEqual(result, SYSTEM_PROMPTS["planner"])

    def test_get_system_prompt_all_known_roles(self):
        """get_system_prompt() should return correct prompts for all roles."""
        for role in SYSTEM_PROMPTS:
            with self.subTest(role=role):
                self.assertEqual(get_system_prompt(role), SYSTEM_PROMPTS[role])

    def test_get_system_prompt_unknown_role_defaults_to_coder(self):
        """An unrecognised role should fall back to the 'coder' system prompt."""
        result = get_system_prompt("totally_unknown_role_xyz")
        self.assertEqual(result, SYSTEM_PROMPTS["coder"])

    def test_get_system_prompt_empty_string_defaults_to_coder(self):
        """An empty-string role should also fall back to the 'coder' prompt."""
        result = get_system_prompt("")
        self.assertEqual(result, SYSTEM_PROMPTS["coder"])


# ===========================================================================
# TestRenderPrompt
# ===========================================================================


class TestRenderPrompt(unittest.TestCase):
    """Tests for render_prompt() template rendering and caching."""

    def setUp(self):
        """Clear global cache before every test to avoid cross-test pollution."""
        clear_prompt_cache()

    def tearDown(self):
        """Ensure cache is clean after each test."""
        clear_prompt_cache()

    # ------------------------------------------------------------------
    # Basic rendering
    # ------------------------------------------------------------------

    def test_render_planner_prompt(self):
        """render_prompt with the planner template should include the goal text
        and the planner system prompt."""
        result = render_prompt("planner", "planner", _PLANNER_PARAMS)
        self.assertIn(_PLANNER_PARAMS["goal"], result)
        self.assertIn(SYSTEM_PROMPTS["planner"], result)

    def test_render_planner_prompt_contains_memory(self):
        """Planner render should embed the memory param."""
        result = render_prompt("planner", "planner", _PLANNER_PARAMS)
        self.assertIn(_PLANNER_PARAMS["memory"], result)

    def test_render_critic_prompt(self):
        """render_prompt with the critic template should include task text and
        the critic system prompt."""
        result = render_prompt("critic", "critic", _CRITIC_PARAMS)
        self.assertIn(_CRITIC_PARAMS["task"], result)
        self.assertIn(SYSTEM_PROMPTS["critic"], result)

    def test_render_coder_prompt(self):
        """render_prompt with the coder template should include the task and
        the coder system prompt."""
        result = render_prompt("coder", "coder", _CODER_PARAMS)
        self.assertIn(_CODER_PARAMS["task"], result)
        self.assertIn(SYSTEM_PROMPTS["coder"], result)

    def test_render_result_is_string(self):
        """render_prompt should always return a str."""
        result = render_prompt("planner", "planner", _PLANNER_PARAMS)
        self.assertIsInstance(result, str)

    def test_render_system_prompt_injected_for_non_default_role(self):
        """The system_prompt placeholder must be replaced (not left literal)."""
        result = render_prompt("planner", "planner", _PLANNER_PARAMS)
        self.assertNotIn("{system_prompt}", result)

    # ------------------------------------------------------------------
    # Unknown template
    # ------------------------------------------------------------------

    def test_render_unknown_template_raises(self):
        """render_prompt with an unknown template_name should raise ValueError."""
        with self.assertRaises(ValueError):
            render_prompt("nonexistent_template", "coder", {"a": "b"})

    # ------------------------------------------------------------------
    # Caching behaviour
    # ------------------------------------------------------------------

    def test_render_uses_cache_on_second_call(self):
        """The second render_prompt call with identical params should be a cache hit."""
        render_prompt("planner", "planner", _PLANNER_PARAMS)  # miss → set
        render_prompt("planner", "planner", _PLANNER_PARAMS)  # hit

        stats = get_cached_prompt_stats()
        self.assertGreaterEqual(stats["hits"], 1)

    def test_render_first_call_is_cache_miss(self):
        """The very first render_prompt call should register as a cache miss."""
        render_prompt("planner", "planner", _PLANNER_PARAMS)
        stats = get_cached_prompt_stats()
        self.assertGreaterEqual(stats["misses"], 1)

    def test_render_no_cache_bypasses_cache(self):
        """render_prompt with use_cache=False should not store or retrieve from cache."""
        render_prompt("planner", "planner", _PLANNER_PARAMS, use_cache=False)
        render_prompt("planner", "planner", _PLANNER_PARAMS, use_cache=False)

        stats = get_cached_prompt_stats()
        # No hits or misses should be recorded when caching is disabled.
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["size"], 0)

    def test_render_cached_result_matches_uncached(self):
        """Cached and uncached renders of the same prompt must be identical."""
        uncached = render_prompt("coder", "coder", _CODER_PARAMS, use_cache=False)
        cached_first = render_prompt("coder", "coder", _CODER_PARAMS, use_cache=True)
        cached_second = render_prompt("coder", "coder", _CODER_PARAMS, use_cache=True)

        self.assertEqual(uncached, cached_first)
        self.assertEqual(cached_first, cached_second)

    def test_render_different_params_produce_different_outputs(self):
        """Different goal texts should yield different rendered strings."""
        params_a = {**_PLANNER_PARAMS, "goal": "Goal A"}
        params_b = {**_PLANNER_PARAMS, "goal": "Goal B"}

        result_a = render_prompt("planner", "planner", params_a, use_cache=False)
        result_b = render_prompt("planner", "planner", params_b, use_cache=False)

        self.assertNotEqual(result_a, result_b)


# ===========================================================================
# TestGlobalCacheHelpers
# ===========================================================================


class TestGlobalCacheHelpers(unittest.TestCase):
    """Tests for get_cached_prompt_stats() and clear_prompt_cache()."""

    def setUp(self):
        clear_prompt_cache()

    def tearDown(self):
        clear_prompt_cache()

    def test_get_cached_prompt_stats_returns_dict(self):
        """get_cached_prompt_stats() should return a dict."""
        stats = get_cached_prompt_stats()
        self.assertIsInstance(stats, dict)

    def test_get_cached_prompt_stats_keys(self):
        """Stats dict should contain the expected keys."""
        stats = get_cached_prompt_stats()
        for key in ("hits", "misses", "hit_rate", "size", "max_size"):
            self.assertIn(key, stats)

    def test_clear_prompt_cache_resets_stats(self):
        """clear_prompt_cache() should reset hit/miss counters and empty cache."""
        render_prompt("planner", "planner", _PLANNER_PARAMS)
        render_prompt("planner", "planner", _PLANNER_PARAMS)  # generates a hit

        clear_prompt_cache()

        stats = get_cached_prompt_stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["size"], 0)


# ===========================================================================
# TestLegacyAliases
# ===========================================================================


class TestLegacyAliases(unittest.TestCase):
    """Tests verifying backward-compatibility aliases equal the originals."""

    def test_planner_legacy_alias_matches_original(self):
        """PLANNER_COT_PROMPT_TEMPLATE must be identical to PLANNER_PROMPT_TEMPLATE."""
        self.assertIs(PLANNER_COT_PROMPT_TEMPLATE, PLANNER_PROMPT_TEMPLATE)

    def test_critic_legacy_alias_matches_original(self):
        """CRITIC_COT_PROMPT_TEMPLATE must be identical to CRITIC_PROMPT_TEMPLATE."""
        self.assertIs(CRITIC_COT_PROMPT_TEMPLATE, CRITIC_PROMPT_TEMPLATE)

    def test_coder_legacy_alias_matches_original(self):
        """CODER_COT_PROMPT_TEMPLATE must be identical to CODER_PROMPT_TEMPLATE."""
        self.assertIs(CODER_COT_PROMPT_TEMPLATE, CODER_PROMPT_TEMPLATE)


if __name__ == "__main__":
    unittest.main()
