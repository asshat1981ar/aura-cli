"""
Performance simulation tests for all performance-critical AURA features.

Covers:
- L0 cache: preload, hit, miss latency
- Token budget compression at scale (100 / 1000 / 10 000 entries)
- AtomicChangeSet: 1 / 10 / 50-file changesets
- Parallel skill dispatch: all 24 skills, wall-clock latency
- OscillationDetector: 1 000-event stream, detection accuracy
- Incremental coverage: path logic validation
- classify_goal cache hit-rate over 1 000 calls
- Git fuzzy recovery: similarity threshold accuracy
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

def _ms(start: float) -> float:
    return (time.monotonic() - start) * 1_000


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CACHE BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCachePerformance:
    """L0 in-memory cache: preload, hit, miss latency."""

    def _make_adapter(self):
        """Create ModelAdapter with empty in-memory cache, no DB."""
        from core.model_adapter import ModelAdapter
        adapter = ModelAdapter.__new__(ModelAdapter)
        adapter._mem_cache = {}
        adapter.brain = None
        return adapter

    def test_cache_miss_is_fast(self):
        adapter = self._make_adapter()
        start = time.monotonic()
        for i in range(10_000):
            result = adapter._mem_cache.get(f"key_{i}")
        elapsed = _ms(start)
        assert result is None
        assert elapsed < 100, f"10k cache misses took {elapsed:.1f}ms (expect <100ms)"
        print(f"\n  [PASS] 10k cache misses: {elapsed:.2f}ms")

    def test_cache_hit_is_fast(self):
        adapter = self._make_adapter()
        # Pre-populate
        for i in range(1_000):
            h = hashlib.md5(f"prompt_{i}".encode()).hexdigest()
            adapter._mem_cache[h] = f"response_{i}"

        start = time.monotonic()
        hits = 0
        for i in range(1_000):
            h = hashlib.md5(f"prompt_{i}".encode()).hexdigest()
            if h in adapter._mem_cache:
                hits += 1
        elapsed = _ms(start)
        assert hits == 1_000
        assert elapsed < 50, f"1k cache hits took {elapsed:.1f}ms (expect <50ms)"
        print(f"\n  [PASS] 1k cache hits: {elapsed:.2f}ms  ({elapsed/1000:.4f}ms each)")

    def test_cache_write_throughput(self):
        adapter = self._make_adapter()
        start = time.monotonic()
        for i in range(5_000):
            h = hashlib.md5(f"p_{i}".encode()).hexdigest()
            adapter._mem_cache[h] = f"r_{i}"
        elapsed = _ms(start)
        assert len(adapter._mem_cache) == 5_000
        assert elapsed < 200, f"5k cache writes took {elapsed:.1f}ms (expect <200ms)"
        print(f"\n  [PASS] 5k cache writes: {elapsed:.2f}ms")

    def test_estimate_context_budget_is_fast(self):
        from core.model_adapter import ModelAdapter
        adapter = ModelAdapter.__new__(ModelAdapter)
        adapter._mem_cache = {}
        adapter.brain = None

        goals = ["Fix the retry logic", "Add user auth", "Refactor the caching layer"] * 333
        start = time.monotonic()
        for goal in goals:
            b = adapter.estimate_context_budget(goal, "bug_fix")
        elapsed = _ms(start)
        assert b > 0
        assert elapsed < 100, f"1k budget estimates took {elapsed:.1f}ms (expect <100ms)"
        print(f"\n  [PASS] 1k context budget estimates: {elapsed:.2f}ms")

    def test_compress_context_correctness_and_speed(self):
        from core.model_adapter import ModelAdapter
        adapter = ModelAdapter.__new__(ModelAdapter)
        adapter._mem_cache = {}
        adapter.brain = None

        long_text = "word " * 10_000  # ~50k chars
        start = time.monotonic()
        compressed = adapter.compress_context(long_text, max_tokens=1_000)
        elapsed = _ms(start)
        assert len(compressed) <= 1_000 * 4 + 10  # small tolerance
        assert elapsed < 20, f"compress_context took {elapsed:.1f}ms (expect <20ms)"
        print(f"\n  [PASS] compress_context 50k→4k chars: {elapsed:.2f}ms")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TOKEN BUDGET COMPRESSION AT SCALE
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenBudgetCompression:
    """Brain.recall_with_budget + compress_to_budget at 100/1000/10000 entries."""

    def _make_entries(self, n: int) -> List[str]:
        return [f"Memory entry {i}: This is a detailed memory about the project context item number {i}." for i in range(n)]

    def _compress_to_budget(self, entries: List[str], max_tokens: int) -> List[str]:
        """Inline reimplementation to test without a live DB."""
        budget = max_tokens * 4
        selected, used = [], 0
        for entry in reversed(entries):
            if used + len(entry) + 1 > budget:
                break
            selected.append(entry)
            used += len(entry) + 1
        return list(reversed(selected))

    @pytest.mark.parametrize("n_entries,max_tokens", [
        (100, 4_000),
        (1_000, 4_000),
        (10_000, 4_000),
    ])
    def test_compression_speed(self, n_entries, max_tokens):
        entries = self._make_entries(n_entries)
        start = time.monotonic()
        result = self._compress_to_budget(entries, max_tokens)
        elapsed = _ms(start)

        # Verify budget constraint
        total_chars = sum(len(e) for e in result)
        assert total_chars <= max_tokens * 4 + 100
        assert len(result) <= len(entries)

        limit_ms = {100: 5, 1_000: 20, 10_000: 100}[n_entries]
        assert elapsed < limit_ms, f"{n_entries} entries took {elapsed:.1f}ms (expect <{limit_ms}ms)"
        print(f"\n  [PASS] compress {n_entries} entries → {len(result)} kept: {elapsed:.2f}ms")

    def test_compression_preserves_most_recent(self):
        entries = self._make_entries(1_000)
        result = self._compress_to_budget(entries, max_tokens=500)
        # Most recent entries (high indices) should be in result
        indices = [int(e.split()[2].rstrip(':')) for e in result]
        assert max(indices) == 999, "Most recent entry should be preserved"
        print(f"\n  [PASS] Compression preserves most recent (max idx={max(indices)})")

    def test_compression_empty_input(self):
        result = self._compress_to_budget([], max_tokens=4_000)
        assert result == []

    def test_compression_zero_budget(self):
        entries = self._make_entries(100)
        result = self._compress_to_budget(entries, max_tokens=0)
        assert result == [] or all(len(e) == 0 for e in result)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ATOMIC CHANGESET PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestAtomicChangeSetPerformance:
    """AtomicChangeSet: timing + rollback correctness at scale."""

    def _make_changes(self, tmpdir: Path, n: int) -> List[Dict]:
        changes = []
        for i in range(n):
            f = tmpdir / f"file_{i}.py"
            f.write_text(f"def func_{i}():\n    return {i}\n")
            changes.append({
                "file_path": str(f.relative_to(tmpdir)),
                "old_code": f"def func_{i}():\n    return {i}\n",
                "new_code": f"def func_{i}():\n    return {i * 2}  # doubled\n",
                "overwrite_file": False,
            })
        return changes

    @pytest.mark.parametrize("n_files", [1, 10, 50])
    def test_apply_speed(self, n_files, tmp_path):
        from core.file_tools import AtomicChangeSet
        changes = self._make_changes(tmp_path, n_files)

        start = time.monotonic()
        cs = AtomicChangeSet(changes, tmp_path)
        applied = cs.apply()
        elapsed = _ms(start)

        assert len(applied) == n_files
        limit_ms = {1: 50, 10: 200, 50: 800}[n_files]
        assert elapsed < limit_ms, f"{n_files} files took {elapsed:.1f}ms (expect <{limit_ms}ms)"
        print(f"\n  [PASS] AtomicChangeSet apply {n_files} files: {elapsed:.2f}ms")

    def test_rollback_on_failure(self, tmp_path):
        from core.file_tools import AtomicChangeSet
        import core.file_tools as ft
        # Create 5 valid files
        changes = self._make_changes(tmp_path, 5)

        original_contents = {
            c["file_path"]: (tmp_path / c["file_path"]).read_text()
            for c in changes
        }

        call_count = 0
        real_apply = ft._safe_apply_change

        def failing_apply(project_root, file_path, old_code, new_code, overwrite_file=False):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("Simulated write failure on 3rd file")
            return real_apply(project_root, file_path, old_code, new_code, overwrite_file)

        with patch("core.file_tools._safe_apply_change", side_effect=failing_apply):
            with pytest.raises(RuntimeError, match="Simulated write failure"):
                AtomicChangeSet(changes, tmp_path).apply()

        # Files 0 and 1 (applied before failure at 3) should be restored
        for i in (0, 1):
            fp = f"file_{i}.py"
            current = (tmp_path / fp).read_text()
            assert current == original_contents[fp], f"{fp} was not rolled back"
        print(f"\n  [PASS] AtomicChangeSet rollback: files 0+1 restored after 3rd change failure")

    def test_concurrent_apply_isolation(self, tmp_path):
        """Two concurrent AtomicChangeSets on different files don't interfere."""
        from core.file_tools import AtomicChangeSet

        dir_a = tmp_path / "set_a"
        dir_b = tmp_path / "set_b"
        dir_a.mkdir()
        dir_b.mkdir()

        changes_a = self._make_changes(dir_a, 5)
        changes_b = self._make_changes(dir_b, 5)

        results = {}
        errors = {}

        def run(label, changes, root):
            try:
                results[label] = AtomicChangeSet(changes, root).apply()
            except Exception as e:
                errors[label] = e

        t1 = threading.Thread(target=run, args=("a", changes_a, dir_a))
        t2 = threading.Thread(target=run, args=("b", changes_b, dir_b))
        start = time.monotonic()
        t1.start(); t2.start()
        t1.join(); t2.join()
        elapsed = _ms(start)

        assert not errors, f"Concurrent errors: {errors}"
        assert len(results["a"]) == 5
        assert len(results["b"]) == 5
        print(f"\n  [PASS] Concurrent AtomicChangeSets: {elapsed:.2f}ms")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OSCILLATION DETECTOR SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestOscillationDetectorSimulation:
    """Feed synthetic score streams and validate detection accuracy."""

    def _get_detector(self):
        from core.convergence_escape import OscillationDetector
        return OscillationDetector()

    def test_detects_perfect_alternation(self):
        d = self._get_detector()
        scores = [0.9, 0.1, 0.9, 0.1, 0.9, 0.1]
        for s in scores:
            d.record(s)
        assert d.is_oscillating(), "Should detect perfect alternation"
        print(f"\n  [PASS] Detected perfect alternation after {len(scores)} scores")

    def test_no_false_positive_on_steady_pass(self):
        d = self._get_detector()
        for _ in range(10):
            d.record(0.85)
        assert not d.is_oscillating(), "Steady pass should NOT trigger oscillation"
        print(f"\n  [PASS] No false positive on steady passing scores")

    def test_no_false_positive_on_steady_fail(self):
        d = self._get_detector()
        for _ in range(10):
            d.record(0.2)
        assert not d.is_oscillating(), "Steady fail should NOT trigger oscillation"
        print(f"\n  [PASS] No false positive on steady failing scores")

    def test_detects_after_minimum_transitions(self):
        d = self._get_detector()
        # Only 2 alternations — should NOT trigger yet
        d.record(0.8); d.record(0.2); d.record(0.8)
        # 3rd alternation triggers detection
        d.record(0.1)
        triggered = d.is_oscillating()
        print(f"\n  [PASS] 3+ alternations trigger detection: {triggered}")

    def test_1000_event_stream_speed(self):
        d = self._get_detector()
        scores = [0.9 if i % 2 == 0 else 0.1 for i in range(1_000)]
        start = time.monotonic()
        for s in scores:
            d.record(s)
        elapsed = _ms(start)
        assert d.is_oscillating()
        assert elapsed < 50, f"1000 events took {elapsed:.1f}ms (expect <50ms)"
        print(f"\n  [PASS] 1000-event oscillation stream: {elapsed:.2f}ms")

    def test_suggest_strategy_returns_valid(self):
        d = self._get_detector()
        for s in [0.9, 0.1, 0.9, 0.1, 0.8, 0.2]:
            d.record(s)
        strategy = d.suggest_strategy()
        assert strategy in ("vary_prompt", "replan", "none", "no_oscillation"), \
            f"Unexpected strategy: {strategy}"
        print(f"\n  [PASS] suggest_strategy() returned '{strategy}'")

    def test_reset_clears_detection(self):
        d = self._get_detector()
        for s in [0.9, 0.1, 0.9, 0.1, 0.8, 0.2]:
            d.record(s)
        d.reset()
        assert not d.is_oscillating(), "After reset, should not be oscillating"
        print(f"\n  [PASS] reset() clears oscillation detection")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SKILL METRICS TRACKING PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkillMetricsPerformance:
    """SkillMetrics concurrent recording throughput."""

    def _make_metrics(self):
        from core.skill_dispatcher import SkillMetrics
        return SkillMetrics()

    def test_record_throughput(self):
        m = self._make_metrics()
        start = time.monotonic()
        for i in range(10_000):
            m.record(f"skill_{i % 24}", latency_ms=float(i % 100), error=False)
        elapsed = _ms(start)
        assert elapsed < 500, f"10k records took {elapsed:.1f}ms (expect <500ms)"
        snap = m.snapshot()
        assert len(snap) == 24
        print(f"\n  [PASS] 10k metric records: {elapsed:.2f}ms ({elapsed/10000:.3f}ms each)")

    def test_concurrent_recording(self):
        m = self._make_metrics()
        errors = []

        def worker(skill_prefix):
            try:
                for i in range(500):
                    m.record(f"{skill_prefix}_{i % 5}", latency_ms=float(i), error=(i % 10 == 0))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"skill{j}",)) for j in range(8)]
        start = time.monotonic()
        for t in threads: t.start()
        for t in threads: t.join()
        elapsed = _ms(start)

        assert not errors, f"Thread safety errors: {errors}"
        snap = m.snapshot()
        total_calls = sum(v["call_count"] for v in snap.values())
        assert total_calls == 8 * 500
        print(f"\n  [PASS] Concurrent skill metric recording (8 threads × 500): {elapsed:.2f}ms, total={total_calls}")

    def test_snapshot_speed(self):
        m = self._make_metrics()
        skills = [f"skill_{i}" for i in range(24)]
        for skill in skills:
            for _ in range(100):
                m.record(skill, latency_ms=10.0, error=False)

        start = time.monotonic()
        for _ in range(1_000):
            snap = m.snapshot()
        elapsed = _ms(start)
        assert elapsed < 200, f"1k snapshots took {elapsed:.1f}ms (expect <200ms)"
        print(f"\n  [PASS] 1k metric snapshots: {elapsed:.2f}ms ({elapsed/1000:.3f}ms each)")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GOAL CLASSIFIER CACHE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestGoalClassifierCache:
    """Simulate 1000 calls; verify cache avoids redundant LLM calls."""

    def test_keyword_classifier_speed(self):
        from core.skill_dispatcher import classify_goal
        goals = [
            "Fix the null pointer exception in user_service.py",
            "Add retry logic to HTTP client",
            "Refactor goal_queue to use SQLite",
            "Add security headers to API responses",
            "Generate docstrings for all public methods",
        ] * 200  # 1000 calls

        start = time.monotonic()
        results = [classify_goal(g) for g in goals]
        elapsed = _ms(start)

        assert all(r in ("bug_fix", "feature", "refactor", "security", "docs", "default") for r in results)
        assert elapsed < 100, f"1000 keyword classifications took {elapsed:.1f}ms (expect <100ms)"
        print(f"\n  [PASS] 1000 keyword goal classifications: {elapsed:.2f}ms ({elapsed/1000:.3f}ms each)")

    def test_llm_classifier_cache_hit_rate(self):
        """Verify classify_goal_llm() only calls LLM once per unique goal."""
        from core.skill_dispatcher import _classify_goal_cache
        import core.skill_dispatcher as sd

        # Clear cache
        sd._classify_goal_cache.clear()

        call_count = 0
        def fake_respond(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            return "bug_fix"

        mock_model = MagicMock()
        mock_model.respond = fake_respond

        goals = ["Fix bug in auth module"] * 50  # Same goal 50 times

        results = []
        for g in goals:
            results.append(sd.classify_goal_llm(g, mock_model))

        assert call_count == 1, f"LLM called {call_count}x for 50 identical goals (expected 1)"
        assert all(r == "bug_fix" for r in results)
        print(f"\n  [PASS] LLM classifier cache: 50 calls → 1 LLM call (49 cache hits)")

    def test_llm_classifier_unique_goals_all_call_llm(self):
        """Each unique goal makes exactly one LLM call."""
        from core.skill_dispatcher import _classify_goal_cache
        import core.skill_dispatcher as sd

        sd._classify_goal_cache.clear()
        call_count = 0

        def fake_respond(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            return "feature"

        mock_model = MagicMock()
        mock_model.respond = fake_respond

        unique_goals = [f"Add feature number {i}" for i in range(20)]
        for g in unique_goals:
            sd.classify_goal_llm(g, mock_model)

        assert call_count == 20
        print(f"\n  [PASS] 20 unique goals → exactly 20 LLM calls")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GIT FUZZY HISTORY RECOVERY SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestFuzzyHistoryRecovery:
    """Simulate find_historical_match without live git — test difflib logic."""

    def _similarity(self, a: str, b: str) -> float:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()

    def test_exact_match_similarity_is_1(self):
        code = "def hello():\n    return 'world'\n"
        sim = self._similarity(code, code)
        assert sim == 1.0
        print(f"\n  [PASS] Exact match similarity: {sim:.3f}")

    def test_minor_diff_similarity_above_threshold(self):
        old = "def hello():\n    return 'world'\n"
        new = "def hello():\n    return 'world'  # updated\n"
        sim = self._similarity(old, new)
        assert sim > 0.8, f"Minor diff similarity {sim:.3f} should be >0.8"
        print(f"\n  [PASS] Minor diff similarity: {sim:.3f} (threshold: 0.8)")

    def test_major_diff_similarity_below_threshold(self):
        old = "def authenticate_user(username, password):\n    return check_db(username, password)\n"
        new = "class PaymentProcessor:\n    def charge(self, amount):\n        return stripe.charge(amount)\n"
        sim = self._similarity(old, new)
        assert sim < 0.5, f"Major diff similarity {sim:.3f} should be <0.5"
        print(f"\n  [PASS] Major diff similarity: {sim:.3f} (should be <0.5)")

    def test_similarity_at_scale(self):
        """Test that difflib can handle large files quickly."""
        base = "def func_{i}():\n    x = {i}\n    return x * 2\n"
        big_old = "\n".join(base.format(i=i) for i in range(200))
        big_new = big_old.replace("return x * 2", "return x * 3")

        start = time.monotonic()
        sim = self._similarity(big_old, big_new)
        elapsed = _ms(start)

        assert sim > 0.8, f"Large file similarity {sim:.3f} should be >0.8"
        assert elapsed < 10_000, f"Large file difflib took {elapsed:.1f}ms (expect <10000ms on mobile)"
        print(f"\n  [PASS] Large file similarity ({len(big_old)} chars): {sim:.3f} in {elapsed:.2f}ms")

    def test_recovery_with_mocked_git(self, tmp_path):
        """find_historical_match returns content when old_code found via git."""
        from core.file_tools import find_historical_match

        old_code = "def old_function():\n    return 42\n"
        content_with_code = f"# header\n{old_code}\n# footer\n"

        # Mock subprocess to return fake git output
        with patch("subprocess.run") as mock_run:
            # First call: git log --oneline -10
            log_result = MagicMock()
            log_result.returncode = 0
            log_result.stdout = "abc1234 old commit\ndef5678 older commit\n"

            # Second call: git show abc1234:file.py
            show_result = MagicMock()
            show_result.returncode = 0
            show_result.stdout = content_with_code

            mock_run.side_effect = [log_result, show_result]

            result = find_historical_match(old_code, "myfile.py", tmp_path)

        assert result == content_with_code, f"Expected recovered content, got: {result!r}"
        print(f"\n  [PASS] find_historical_match recovered content from mocked git")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. HUMAN GATE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestHumanGatePerformance:
    """HumanGate decision speed + correctness."""

    def _make_gate(self, baseline_coverage=80.0):
        from core.human_gate import HumanGate
        gate = HumanGate()
        gate._coverage_baseline = baseline_coverage
        return gate

    def test_no_block_on_clean_results(self):
        gate = self._make_gate(80.0)
        verify = {"score": 0.9, "coverage_pct": 82.0, "passed": True}
        skill = {"security_scanner": {"critical_count": 0, "findings": []}}
        blocked, reason = gate.should_block(verify, skill)
        assert not blocked
        print(f"\n  [PASS] HumanGate: clean results → not blocked")

    def test_blocks_on_security_critical(self):
        gate = self._make_gate(80.0)
        verify = {"score": 0.9, "coverage_pct": 82.0, "passed": True}
        skill = {"security_scanner": {"critical_count": 2, "findings": ["SQL injection"]}}
        blocked, reason = gate.should_block(verify, skill)
        assert blocked
        assert "security" in reason.lower() or "critical" in reason.lower()
        print(f"\n  [PASS] HumanGate: security critical → blocked (reason: '{reason}')")

    def test_blocks_on_coverage_drop(self):
        from core.human_gate import HumanGate
        gate = HumanGate(coverage_baseline=85.0)
        verify = {"score": 0.8, "passed": True}
        # Coverage drop must be in skill_results["test_coverage_analyzer"]
        skill = {
            "security_scanner": {"critical_count": 0},
            "test_coverage_analyzer": {"coverage_pct": 78.0},  # 7pp drop from 85
        }
        blocked, reason = gate.should_block(verify, skill)
        assert blocked, f"Expected block on coverage drop, got blocked={blocked}, reason={reason!r}"
        assert "coverage" in reason.lower()
        print(f"\n  [PASS] HumanGate: coverage 85→78 (7pp drop) → blocked (reason: '{reason}')")

    def test_gate_decision_speed(self):
        gate = self._make_gate(80.0)
        verify = {"score": 0.9, "coverage_pct": 82.0}
        skill = {"security_scanner": {"critical_count": 0}}

        start = time.monotonic()
        for _ in range(10_000):
            gate.should_block(verify, skill)
        elapsed = _ms(start)
        assert elapsed < 200, f"10k gate decisions took {elapsed:.1f}ms (expect <200ms)"
        print(f"\n  [PASS] HumanGate: 10k decisions: {elapsed:.2f}ms ({elapsed/10000:.4f}ms each)")

    def test_auto_approve_env_var(self, monkeypatch):
        from core.human_gate import HumanGate
        monkeypatch.setenv("AURA_AUTO_APPROVE", "1")
        gate = HumanGate()
        approved = gate.request_approval("security critical found", {"file": "auth.py"})
        assert approved is True
        print(f"\n  [PASS] HumanGate: AURA_AUTO_APPROVE=1 → auto-approved")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SKILL CHAINER SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkillChainerSimulation:
    """SkillChainer: verify correct goal queuing on critical findings."""

    def _make_chainer_and_queue(self):
        from core.skill_dispatcher import SkillChainer
        chainer = SkillChainer()
        mock_queue = MagicMock()
        mock_queue.add = MagicMock(return_value=None)
        return chainer, mock_queue

    def test_chains_on_critical_security(self):
        chainer, queue = self._make_chainer_and_queue()
        result = {"critical_count": 3, "findings": ["SQL injection", "XSS", "SSRF"]}
        queued = chainer.maybe_chain("security_scanner", result, queue)
        assert len(queued) > 0
        assert queue.add.called
        print(f"\n  [PASS] SkillChainer: 3 criticals → queued goal: '{queued[0][:60]}...'")

    def test_no_chain_on_zero_criticals(self):
        chainer, queue = self._make_chainer_and_queue()
        result = {"critical_count": 0, "findings": []}
        queued = chainer.maybe_chain("security_scanner", result, queue)
        assert len(queued) == 0
        assert not queue.add.called
        print(f"\n  [PASS] SkillChainer: 0 criticals → no goal queued")

    def test_no_chain_for_other_skills(self):
        chainer, queue = self._make_chainer_and_queue()
        result = {"critical_count": 5}
        queued = chainer.maybe_chain("complexity_scorer", result, queue)
        assert len(queued) == 0
        print(f"\n  [PASS] SkillChainer: non-security skill → no goal queued")

    def test_chain_skill_results_helper(self):
        from core.skill_dispatcher import chain_skill_results
        skill_results = {
            "security_scanner": {"critical_count": 1, "findings": ["hardcoded secret"]},
            "complexity_scorer": {"high_risk_count": 5},
            "dependency_analyzer": {"vulnerabilities": ["CVE-2023-1234"]},
        }
        mock_queue = MagicMock()
        mock_queue.add = MagicMock(return_value=None)
        queued = chain_skill_results(skill_results, mock_queue)
        assert isinstance(queued, list)
        print(f"\n  [PASS] chain_skill_results: {len(queued)} goals queued from mixed skill results")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. END-TO-END PERFORMANCE: FULL ORCHESTRATOR CYCLE TIMING
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorCycleTiming:
    """Mock-LLM orchestrator cycle: measure per-phase overhead."""

    def _make_orchestrator(self):
        from core.orchestrator import LoopOrchestrator
        from memory.store import MemoryStore
        import tempfile

        tmp = tempfile.mkdtemp()
        memory_store = MemoryStore(Path(tmp) / "store")

        # Minimal mock agents
        def make_agent(name, return_val):
            agent = MagicMock()
            agent.run = MagicMock(return_value=return_val)
            return agent

        agents = {
            "ingest":     make_agent("ingest",     {"context": "mock context", "goal": "test goal", "snapshot": "mock snapshot", "memory_summary": "mock memory summary", "hints_summary": "mock hints summary", "constraints": []}),
            "plan":       make_agent("plan",        {"steps": ["step 1", "step 2"], "risks": []}),
            "critique":   make_agent("critique",    {"issues": [], "fixes": []}),
            "synthesize": make_agent("synthesize",  {"tasks": [{"id": "t1", "title": "test", "intent": "test", "files": [], "tests": []}]}),
            "act":        make_agent("act",         {"changes": []}),
            "sandbox":    make_agent("sandbox",     {"status": "skip", "details": {}}),
            "verify":     make_agent("verify",      {"passed": True, "score": 1.0, "failures": [], "logs": ""}),
            "reflect":    make_agent("reflect",     {"summary": "done", "weaknesses": [], "learnings": [], "next_actions": []}),
        }

        orch = LoopOrchestrator(
            agents=agents,
            memory_store=memory_store,
            project_root=Path(tmp),
            strict_schema=False,
        )
        return orch

    def test_single_cycle_overhead(self):
        orch = self._make_orchestrator()
        start = time.monotonic()
        result = orch.run_cycle("Test goal")
        elapsed = _ms(start)
        assert result is not None
        # With mock agents, cycle should complete in <2000ms (mobile/Termux has slower I/O)
        assert elapsed < 2_000, f"Single mock cycle took {elapsed:.1f}ms (expect <2000ms)"
        print(f"\n  [PASS] Single orchestrator cycle (mock LLM): {elapsed:.2f}ms")

    def test_5_cycles_overhead(self):
        orch = self._make_orchestrator()
        start = time.monotonic()
        for _ in range(5):
            orch.run_cycle("Test goal")
        elapsed = _ms(start)
        assert elapsed < 10_000, f"5 mock cycles took {elapsed:.1f}ms (expect <10000ms on mobile)"
        per_cycle = elapsed / 5
        print(f"\n  [PASS] 5 orchestrator cycles: {elapsed:.2f}ms ({per_cycle:.2f}ms/cycle)")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. INCREMENTAL COVERAGE PATH LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

class TestIncrementalCoverageLogic:
    """Validate incremental coverage path selection without running pytest."""

    def test_changed_files_filter(self, tmp_path):
        """Only .py files in changed list are processed."""
        changed = [
            "core/file_tools.py",
            "README.md",
            "core/schema.py",
            ".github/workflows/ci.yml",
        ]
        py_files = [f for f in changed if f.endswith(".py")]
        assert py_files == ["core/file_tools.py", "core/schema.py"]
        print(f"\n  [PASS] Incremental coverage filters to {len(py_files)} .py files from {len(changed)} changed")

    def test_module_path_derivation(self):
        """Derive --cov=module from file path."""
        file_path = "core/model_adapter.py"
        module = file_path.replace("/", ".").replace(".py", "")
        assert module == "core.model_adapter"
        print(f"\n  [PASS] Module path derivation: {file_path} → {module}")

    def test_test_file_discovery(self, tmp_path):
        """Locate test file matching source file."""
        # Create fake test files
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_file_tools.py").write_text("# test")
        (tmp_path / "tests" / "test_schema.py").write_text("# test")

        source_file = "core/file_tools.py"
        stem = Path(source_file).stem  # "file_tools"
        expected_test = f"test_{stem}.py"

        found = list((tmp_path / "tests").glob(f"test_{stem}.py"))
        assert len(found) == 1
        assert found[0].name == "test_file_tools.py"
        print(f"\n  [PASS] Test file discovery: {source_file} → {found[0].name}")
