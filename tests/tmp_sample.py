"""Sample module for AST chunker tests."""

import math

CONSTANT = 42


def greet(name):
    """Return a friendly greeting."""
    return f"hello, {name}"


class Calculator:
    """A simple calculator."""

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
