def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers with error handling for division by zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
