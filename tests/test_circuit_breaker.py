"""Tests for _CircuitBreaker in memory/momento_adapter.py."""
import time
import unittest
from unittest.mock import patch

from memory.momento_adapter import _CircuitBreaker


class TestCircuitBreakerClosed(unittest.TestCase):
    def test_allows_requests_when_closed(self):
        cb = _CircuitBreaker()
        self.assertTrue(cb.allow_request())
        self.assertEqual(cb._state, "closed")

    def test_success_keeps_closed(self):
        cb = _CircuitBreaker()
        cb.record_success()
        self.assertEqual(cb._state, "closed")
        self.assertEqual(cb._failures, 0)


class TestCircuitBreakerOpens(unittest.TestCase):
    def _open_circuit(self, cb: _CircuitBreaker) -> None:
        for _ in range(cb.THRESHOLD):
            cb.record_failure()

    def test_opens_after_threshold_failures(self):
        cb = _CircuitBreaker()
        self._open_circuit(cb)
        self.assertEqual(cb._state, "open")

    def test_blocks_requests_when_open(self):
        cb = _CircuitBreaker()
        self._open_circuit(cb)
        self.assertFalse(cb.allow_request())

    def test_failure_count_below_threshold_stays_closed(self):
        cb = _CircuitBreaker()
        for _ in range(cb.THRESHOLD - 1):
            cb.record_failure()
        self.assertEqual(cb._state, "closed")
        self.assertTrue(cb.allow_request())


class TestCircuitBreakerHalfOpen(unittest.TestCase):
    def _open_circuit(self, cb: _CircuitBreaker) -> None:
        for _ in range(cb.THRESHOLD):
            cb.record_failure()

    def test_transitions_to_half_open_after_reset(self):
        cb = _CircuitBreaker()
        self._open_circuit(cb)
        cb._opened_at = time.monotonic() - cb.RESET_SECONDS - 1
        self.assertTrue(cb.allow_request())
        self.assertEqual(cb._state, "half_open")

    def test_probe_success_closes_circuit(self):
        cb = _CircuitBreaker()
        self._open_circuit(cb)
        cb._opened_at = time.monotonic() - cb.RESET_SECONDS - 1
        cb.allow_request()  # transition to half_open
        cb.record_success()
        self.assertEqual(cb._state, "closed")
        self.assertEqual(cb._failures, 0)

    def test_probe_failure_reopens_circuit(self):
        cb = _CircuitBreaker()
        self._open_circuit(cb)
        cb._opened_at = time.monotonic() - cb.RESET_SECONDS - 1
        cb.allow_request()  # transition to half_open
        cb.record_failure()
        self.assertEqual(cb._state, "open")

    def test_still_blocks_before_reset_window(self):
        cb = _CircuitBreaker()
        self._open_circuit(cb)
        # Don't advance time — should still block
        self.assertFalse(cb.allow_request())
        self.assertEqual(cb._state, "open")
