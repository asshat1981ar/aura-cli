"""Basic tests for utils.py in sample_project fixture."""
import pytest
from .utils import format_name, count_words, is_palindrome, clamp


class TestFormatName:
    def test_basic(self):
        assert format_name("Alice", "Smith") == "Alice Smith"

    def test_strips_whitespace(self):
        assert format_name("  Bob  ", "  Jones  ") == "Bob Jones"


class TestCountWords:
    def test_empty_string(self):
        assert count_words("") == 0

    def test_single_word(self):
        assert count_words("hello") == 1

    def test_multiple_words(self):
        assert count_words("hello world foo") == 3


class TestIsPalindrome:
    def test_simple_palindrome(self):
        assert is_palindrome("racecar") is True

    def test_not_palindrome(self):
        assert is_palindrome("hello") is False

    def test_with_spaces(self):
        assert is_palindrome("a man a plan a canal panama".replace(" ", "")) is True


class TestClamp:
    def test_within_range(self):
        assert clamp(5, 0, 10) == 5

    def test_below_minimum(self):
        assert clamp(-1, 0, 10) == 0

    def test_above_maximum(self):
        assert clamp(15, 0, 10) == 10
