"""Simple arithmetic functions for sample project (fixture for E2E tests).

These functions intentionally lack docstrings and type hints.
"""


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def power(base, exp):
    return base ** exp


def average(numbers):
    if not numbers:
        raise ValueError("Cannot average empty list")
    return sum(numbers) / len(numbers)
