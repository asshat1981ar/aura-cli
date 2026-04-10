"""Basic tests for calculator.py in sample_project fixture."""

import pytest
from .calculator import add, subtract, multiply, divide, power, average


class TestAdd:
    def test_positive(self):
        assert add(2, 3) == 5

    def test_negative(self):
        assert add(-1, -2) == -3

    def test_zero(self):
        assert add(0, 5) == 5


class TestSubtract:
    def test_basic(self):
        assert subtract(10, 4) == 6


class TestMultiply:
    def test_basic(self):
        assert multiply(3, 4) == 12

    def test_by_zero(self):
        assert multiply(5, 0) == 0


class TestDivide:
    def test_basic(self):
        assert divide(10, 2) == 5.0

    def test_divide_by_zero(self):
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(5, 0)


class TestPower:
    def test_basic(self):
        assert power(2, 3) == 8

    def test_zero_exponent(self):
        assert power(5, 0) == 1


class TestAverage:
    def test_basic(self):
        assert average([1, 2, 3, 4, 5]) == 3.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Cannot average empty list"):
            average([])
