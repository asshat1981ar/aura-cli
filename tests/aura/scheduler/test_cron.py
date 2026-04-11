"""Tests for cron expression parser."""

import pytest
from datetime import datetime

from aura.scheduler.cron import CronExpression, is_valid_cron, parse_cron


class TestCronExpression:
    def test_parse_simple(self):
        cron = CronExpression("0 2 * * *")  # 2:00 AM daily

        assert cron.fields[0] == {0}  # minute
        assert cron.fields[1] == {2}  # hour
        assert len(cron.fields[2]) == 31  # all days
        assert len(cron.fields[3]) == 12  # all months
        assert len(cron.fields[4]) == 7  # all weekdays

    def test_parse_every_5_minutes(self):
        cron = CronExpression("*/5 * * * *")

        assert cron.fields[0] == {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}
        assert len(cron.fields[1]) == 24

    def test_parse_range(self):
        cron = CronExpression("0 9-17 * * 1-5")  # Business hours

        assert cron.fields[1] == {9, 10, 11, 12, 13, 14, 15, 16, 17}
        assert cron.fields[4] == {1, 2, 3, 4, 5}

    def test_parse_list(self):
        cron = CronExpression("0 0,12 * * *")  # Midnight and noon

        assert cron.fields[1] == {0, 12}

    def test_parse_step_with_range(self):
        cron = CronExpression("0 0-12/2 * * *")  # Every 2 hours from midnight to noon

        assert cron.fields[1] == {0, 2, 4, 6, 8, 10, 12}

    def test_invalid_expression(self):
        with pytest.raises(ValueError):
            CronExpression("invalid")

        with pytest.raises(ValueError):
            CronExpression("* * *")  # Too few fields

    def test_matches(self):
        cron = CronExpression("30 14 * * *")  # 2:30 PM daily

        dt = datetime(2024, 1, 15, 14, 30, 0)
        assert cron.matches(dt) is True

        dt = datetime(2024, 1, 15, 14, 31, 0)
        assert cron.matches(dt) is False

    def test_matches_weekday(self):
        cron = CronExpression("0 9 * * 1")  # 9:00 AM Mondays

        # Monday, Jan 15, 2024
        dt = datetime(2024, 1, 15, 9, 0, 0)
        assert cron.matches(dt) is True

        # Tuesday, Jan 16, 2024
        dt = datetime(2024, 1, 16, 9, 0, 0)
        assert cron.matches(dt) is False

    def test_next_run(self):
        cron = CronExpression("0 * * * *")  # Every hour

        # Start at 2:30
        after = datetime(2024, 1, 15, 2, 30, 0)
        next_run = cron.next_run(after)

        # Next run should be 3:00
        assert next_run == datetime(2024, 1, 15, 3, 0, 0)

    def test_next_run_same_hour(self):
        cron = CronExpression("30 * * * *")  # Every hour at :30

        # Start at 2:00
        after = datetime(2024, 1, 15, 2, 0, 0)
        next_run = cron.next_run(after)

        # Next run should be 2:30
        assert next_run == datetime(2024, 1, 15, 2, 30, 0)


class TestParseCron:
    def test_parse_cron(self):
        cron = parse_cron("0 0 * * *")
        assert isinstance(cron, CronExpression)

    def test_is_valid_cron_valid(self):
        assert is_valid_cron("0 2 * * *") is True
        assert is_valid_cron("*/5 * * * *") is True

    def test_is_valid_cron_invalid(self):
        assert is_valid_cron("invalid") is False
        assert is_valid_cron("* * *") is False
        assert is_valid_cron("") is False
