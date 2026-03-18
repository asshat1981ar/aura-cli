"""Tests for circuit breaker implementations."""
import time
import unittest
from unittest.mock import patch

from core.circuit_breaker import CircuitBreaker


class TestCircuitBreakerClosed(unittest.TestCase):
    def test_initially_closed(self):
        cb = CircuitBreaker(threshold=3, cooldown_s=10)
        self.assertFalse(cb.is_open())
        self.assertEqual(cb.consecutive_fails, 0)

    def test_success_keeps_closed(self):
        cb = CircuitBreaker()
        cb.record_success()
        self.assertFalse(cb.is_open())
        self.assertEqual(cb.consecutive_fails, 0)


class TestCircuitBreakerOpens(unittest.TestCase):
    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=3, cooldown_s=60)
        cb.record_failure()
        cb.record_failure()
        self.assertFalse(cb.is_open())
        cb.record_failure()
        self.assertTrue(cb.is_open())

    def test_failure_count_below_threshold_stays_closed(self):
        cb = CircuitBreaker(threshold=5, cooldown_s=60)
        for _ in range(4):
            cb.record_failure()
        self.assertFalse(cb.is_open())


class TestCircuitBreakerRecovery(unittest.TestCase):
    def test_success_resets_after_open(self):
        cb = CircuitBreaker(threshold=2, cooldown_s=60)
        cb.record_failure()
        cb.record_failure()
        self.assertTrue(cb.is_open())
        cb.record_success()
        self.assertFalse(cb.is_open())
        self.assertEqual(cb.consecutive_fails, 0)

    def test_cooldown_closes_circuit(self):
        cb = CircuitBreaker(threshold=2, cooldown_s=0.05)
        cb.record_failure()
        cb.record_failure()
        self.assertTrue(cb.is_open())
        time.sleep(0.06)
        self.assertFalse(cb.is_open())
        self.assertEqual(cb.consecutive_fails, 0)

    def test_as_dict(self):
        cb = CircuitBreaker(threshold=5, cooldown_s=60)
        cb.record_failure()
        d = cb.as_dict()
        self.assertEqual(d["consecutive_fails"], 1)
        self.assertEqual(d["threshold"], 5)
        self.assertFalse(d["is_open"])
        self.assertEqual(d["cooldown_s"], 60.0)


# ── Legacy Momento circuit breaker tests ──
# Keep these for backward compatibility if momento_adapter is present
try:
    from memory.momento_adapter import _CircuitBreaker as MomentoCircuitBreaker

    class TestMomentoCircuitBreakerClosed(unittest.TestCase):
        def test_allows_requests_when_closed(self):
            cb = MomentoCircuitBreaker()
            self.assertTrue(cb.allow_request())
            self.assertEqual(cb._state, "closed")

        def test_success_keeps_closed(self):
            cb = MomentoCircuitBreaker()
            cb.record_success()
            self.assertEqual(cb._state, "closed")
            self.assertEqual(cb._failures, 0)

    class TestMomentoCircuitBreakerOpens(unittest.TestCase):
        def _open_circuit(self, cb):
            for _ in range(cb.THRESHOLD):
                cb.record_failure()

        def test_opens_after_threshold_failures(self):
            cb = MomentoCircuitBreaker()
            self._open_circuit(cb)
            self.assertEqual(cb._state, "open")

        def test_blocks_requests_when_open(self):
            cb = MomentoCircuitBreaker()
            self._open_circuit(cb)
            self.assertFalse(cb.allow_request())

        def test_failure_count_below_threshold_stays_closed(self):
            cb = MomentoCircuitBreaker()
            for _ in range(cb.THRESHOLD - 1):
                cb.record_failure()
            self.assertEqual(cb._state, "closed")
            self.assertTrue(cb.allow_request())

    class TestMomentoCircuitBreakerHalfOpen(unittest.TestCase):
        def _open_circuit(self, cb):
            for _ in range(cb.THRESHOLD):
                cb.record_failure()

        def test_transitions_to_half_open_after_reset(self):
            cb = MomentoCircuitBreaker()
            self._open_circuit(cb)
            cb._opened_at = time.monotonic() - cb.RESET_SECONDS - 1
            self.assertTrue(cb.allow_request())
            self.assertEqual(cb._state, "half_open")

        def test_probe_success_closes_circuit(self):
            cb = MomentoCircuitBreaker()
            self._open_circuit(cb)
            cb._opened_at = time.monotonic() - cb.RESET_SECONDS - 1
            cb.allow_request()
            cb.record_success()
            self.assertEqual(cb._state, "closed")
            self.assertEqual(cb._failures, 0)

        def test_probe_failure_reopens_circuit(self):
            cb = MomentoCircuitBreaker()
            self._open_circuit(cb)
            cb._opened_at = time.monotonic() - cb.RESET_SECONDS - 1
            cb.allow_request()
            cb.record_failure()
            self.assertEqual(cb._state, "open")

        def test_still_blocks_before_reset_window(self):
            cb = MomentoCircuitBreaker()
            self._open_circuit(cb)
            self.assertFalse(cb.allow_request())
            self.assertEqual(cb._state, "open")

except ImportError:
    pass  # momento_adapter not available, skip legacy tests
