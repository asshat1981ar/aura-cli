"""Tests for core/policy.py and core/policies/* — Policy, SlidingWindow, TimeBound, ResourceBound."""

import time
import pytest
from unittest.mock import patch

from core.policy import Policy
from core.policies.sliding_window import SlidingWindowPolicy
from core.policies.time_bound import TimeBoundPolicy
from core.policies.resource_bound import ResourceBoundPolicy


# ---------------------------------------------------------------------------
# SlidingWindowPolicy
# ---------------------------------------------------------------------------


class TestSlidingWindowPolicy:
    def test_pass_when_verification_passes(self):
        p = SlidingWindowPolicy(max_cycles=5)
        result = p.evaluate([], {"status": "pass"})
        assert result == "PASS"

    def test_max_cycles_when_history_full(self):
        p = SlidingWindowPolicy(max_cycles=3)
        result = p.evaluate([{}, {}, {}], {"status": "fail"})
        assert result == "MAX_CYCLES"

    def test_empty_string_when_cycles_remaining(self):
        p = SlidingWindowPolicy(max_cycles=5)
        result = p.evaluate([{}], {"status": "fail"})
        assert result == ""

    def test_pass_takes_priority_over_max_cycles(self):
        p = SlidingWindowPolicy(max_cycles=1)
        result = p.evaluate([{}, {}], {"status": "pass"})
        assert result == "PASS"

    def test_default_max_cycles_is_5(self):
        p = SlidingWindowPolicy()
        assert p.max_cycles == 5


# ---------------------------------------------------------------------------
# TimeBoundPolicy
# ---------------------------------------------------------------------------


class TestTimeBoundPolicy:
    def test_pass_when_verification_passes(self):
        p = TimeBoundPolicy(max_seconds=60)
        result = p.evaluate([], {"status": "pass"}, started_at=time.time())
        assert result == "PASS"

    def test_empty_string_when_no_started_at(self):
        p = TimeBoundPolicy(max_seconds=60)
        result = p.evaluate([], {"status": "fail"}, started_at=None)
        assert result == ""

    def test_time_limit_when_expired(self):
        p = TimeBoundPolicy(max_seconds=1)
        result = p.evaluate([], {"status": "fail"}, started_at=time.time() - 2)
        assert result == "TIME_LIMIT"

    def test_empty_when_within_limit(self):
        p = TimeBoundPolicy(max_seconds=60)
        result = p.evaluate([], {"status": "fail"}, started_at=time.time())
        assert result == ""

    def test_default_max_seconds(self):
        p = TimeBoundPolicy()
        assert p.max_seconds == 120


# ---------------------------------------------------------------------------
# ResourceBoundPolicy
# ---------------------------------------------------------------------------


class TestResourceBoundPolicy:
    def test_pass_when_verification_passes(self):
        p = ResourceBoundPolicy(max_tokens=100)
        result = p.evaluate([], {"status": "pass"})
        assert result == "PASS"

    def test_empty_when_under_budget(self):
        p = ResourceBoundPolicy(max_tokens=100000)
        history = [{"phase_outputs": {"act": "short"}}]
        result = p.evaluate(history, {"status": "fail"})
        assert result == ""

    def test_token_budget_exceeded(self):
        p = ResourceBoundPolicy(max_tokens=1)
        # Large phase_outputs to exceed budget
        big = "x" * 100
        history = [{"phase_outputs": big}]
        result = p.evaluate(history, {"status": "fail"})
        assert result == "TOKEN_BUDGET_EXCEEDED"

    def test_default_max_tokens(self):
        p = ResourceBoundPolicy()
        assert p.max_tokens == 50000

    def test_chars_per_token_constant(self):
        assert ResourceBoundPolicy.CHARS_PER_TOKEN == 4


# ---------------------------------------------------------------------------
# Policy (facade)
# ---------------------------------------------------------------------------


class TestPolicyFacade:
    def test_default_uses_sliding_window(self):
        p = Policy()
        assert isinstance(p.impl, SlidingWindowPolicy)

    def test_max_seconds_uses_time_bound(self):
        p = Policy(max_seconds=30)
        assert isinstance(p.impl, TimeBoundPolicy)

    def test_explicit_impl_used(self):
        impl = ResourceBoundPolicy(max_tokens=999)
        p = Policy(impl=impl)
        assert p.impl is impl

    def test_evaluate_delegates_to_impl(self):
        p = Policy(max_cycles=3)
        result = p.evaluate([{}, {}, {}], {"status": "fail"})
        assert result == "MAX_CYCLES"

    def test_from_config_sliding_window(self):
        p = Policy.from_config({"policy_name": "sliding_window", "policy_max_cycles": 7})
        assert isinstance(p.impl, SlidingWindowPolicy)
        assert p.impl.max_cycles == 7

    def test_from_config_time_bound(self):
        p = Policy.from_config({"policy_name": "time_bound", "policy_max_seconds": 90})
        assert isinstance(p.impl, TimeBoundPolicy)
        assert p.impl.max_seconds == 90

    def test_from_config_resource_bound(self):
        p = Policy.from_config({"policy_name": "resource_bound", "policy_max_tokens": 20000})
        assert isinstance(p.impl, ResourceBoundPolicy)
        assert p.impl.max_tokens == 20000

    def test_from_config_default_is_sliding_window(self):
        p = Policy.from_config({})
        assert isinstance(p.impl, SlidingWindowPolicy)
