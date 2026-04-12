"""Tests for core/human_gate.py — HumanGate."""

import pytest
from unittest.mock import patch

from core.human_gate import HumanGate


@pytest.fixture
def gate():
    return HumanGate()


@pytest.fixture
def gate_with_baseline():
    return HumanGate(coverage_baseline=80.0)


# ---------------------------------------------------------------------------
# should_block — security scanner
# ---------------------------------------------------------------------------

class TestShouldBlockSecurity:
    def test_no_critical_findings_no_block(self, gate):
        blocked, reason = gate.should_block(
            {"status": "pass"},
            {"security_scanner": {"critical_count": 0}},
        )
        assert blocked is False
        assert reason == ""

    def test_critical_findings_blocks(self, gate):
        blocked, reason = gate.should_block(
            {"status": "pass"},
            {"security_scanner": {"critical_count": 3}},
        )
        assert blocked is True
        assert "3" in reason
        assert "critical" in reason

    def test_missing_security_key_no_block(self, gate):
        blocked, _ = gate.should_block({}, {})
        assert blocked is False

    def test_non_int_critical_count_no_block(self, gate):
        blocked, _ = gate.should_block({}, {"security_scanner": {"critical_count": "many"}})
        assert blocked is False

    def test_one_critical_finding_blocks(self, gate):
        blocked, _ = gate.should_block({}, {"security_scanner": {"critical_count": 1}})
        assert blocked is True


# ---------------------------------------------------------------------------
# should_block — coverage baseline
# ---------------------------------------------------------------------------

class TestShouldBlockCoverage:
    def test_no_baseline_no_coverage_block(self, gate):
        blocked, _ = gate.should_block(
            {},
            {"test_coverage_analyzer": {"coverage_pct": 10.0}},
        )
        assert blocked is False

    def test_coverage_above_threshold_no_block(self, gate_with_baseline):
        # 80 baseline, drop only 3pp → below threshold of 5
        blocked, _ = gate_with_baseline.should_block(
            {}, {"test_coverage_analyzer": {"coverage_pct": 77.0}}
        )
        assert blocked is False

    def test_coverage_drop_exceeds_threshold_blocks(self, gate_with_baseline):
        # 80 baseline, current 74 → drop of 6pp > 5pp threshold
        blocked, reason = gate_with_baseline.should_block(
            {}, {"test_coverage_analyzer": {"coverage_pct": 74.0}}
        )
        assert blocked is True
        assert "coverage" in reason.lower()
        assert "80.0" in reason

    def test_coverage_exactly_at_threshold_no_block(self, gate_with_baseline):
        # drop exactly 5pp = threshold, not strictly greater
        blocked, _ = gate_with_baseline.should_block(
            {}, {"test_coverage_analyzer": {"coverage_pct": 75.0}}
        )
        assert blocked is False

    def test_missing_coverage_pct_no_block(self, gate_with_baseline):
        blocked, _ = gate_with_baseline.should_block(
            {}, {"test_coverage_analyzer": {}}
        )
        assert blocked is False

    def test_non_numeric_coverage_pct_no_block(self, gate_with_baseline):
        blocked, _ = gate_with_baseline.should_block(
            {}, {"test_coverage_analyzer": {"coverage_pct": "n/a"}}
        )
        assert blocked is False

    def test_security_checked_before_coverage(self, gate_with_baseline):
        # Both conditions present — security blocks first
        blocked, reason = gate_with_baseline.should_block(
            {},
            {
                "security_scanner": {"critical_count": 2},
                "test_coverage_analyzer": {"coverage_pct": 60.0},
            },
        )
        assert blocked is True
        assert "security_scanner" in reason


# ---------------------------------------------------------------------------
# request_approval
# ---------------------------------------------------------------------------

class TestRequestApproval:
    def test_auto_approve_env_var_returns_true(self, gate):
        with patch.dict("os.environ", {"AURA_AUTO_APPROVE": "1"}):
            result = gate.request_approval("risky change", {})
        assert result is True

    def test_non_tty_stdin_returns_false(self, gate):
        with patch("sys.stdin.isatty", return_value=False):
            with patch.dict("os.environ", {}, clear=True):
                result = gate.request_approval("risky change", {})
        assert result is False

    def test_interactive_yes_returns_true(self, gate):
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", return_value="y"):
                with patch.dict("os.environ", {}, clear=True):
                    result = gate.request_approval("risky", {})
        assert result is True

    def test_interactive_yes_full_word(self, gate):
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", return_value="yes"):
                with patch.dict("os.environ", {}, clear=True):
                    result = gate.request_approval("risky", {})
        assert result is True

    def test_interactive_no_returns_false(self, gate):
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", return_value="n"):
                with patch.dict("os.environ", {}, clear=True):
                    result = gate.request_approval("risky", {})
        assert result is False

    def test_interactive_empty_input_returns_false(self, gate):
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", return_value=""):
                with patch.dict("os.environ", {}, clear=True):
                    result = gate.request_approval("risky", {})
        assert result is False

    def test_eof_on_input_returns_false(self, gate):
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", side_effect=EOFError):
                with patch.dict("os.environ", {}, clear=True):
                    result = gate.request_approval("risky", {})
        assert result is False

    def test_auto_approve_other_value_not_approved(self, gate):
        with patch("sys.stdin.isatty", return_value=False):
            with patch.dict("os.environ", {"AURA_AUTO_APPROVE": "true"}):
                result = gate.request_approval("risky", {})
        assert result is False


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestHumanGateInit:
    def test_default_no_baseline(self):
        g = HumanGate()
        assert g.coverage_baseline is None

    def test_baseline_stored(self):
        g = HumanGate(coverage_baseline=75.5)
        assert g.coverage_baseline == 75.5

    def test_threshold_constant(self):
        assert HumanGate.COVERAGE_DROP_THRESHOLD == 5.0
