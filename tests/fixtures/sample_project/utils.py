"""Utility functions for sample project (fixture for E2E tests).

These functions intentionally lack type hints and comprehensive docstrings
so that AURA can be tested adding them.
"""


def format_name(first, last):
    return f"{first.strip()} {last.strip()}"


def count_words(text):
    if not text:
        return 0
    return len(text.split())


def is_palindrome(s):
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]


def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value
