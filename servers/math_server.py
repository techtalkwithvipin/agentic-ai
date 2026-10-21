from mcp.server.fastmcp import FastMCP

mcp= FastMCP("Math")

@mcp.tool()
def add(x: int, y: int) -> int:
    """
    Tool that adds two numbers.
    Args:
        x (int): The first number.
        y (int): The second number.
    Returns:
        int: The sum of the two numbers.
    """
    return x + y

@mcp.tool()
def subtract(x: int, y: int) -> int:
    """
    Tool that subtracts the second number from the first.
    Args:
        x (int): The first number.
        y (int): The second number.
    Returns:
        int: The result of the subtraction.
    """
    return x - y

@mcp.tool()
def multiply(x: int, y: int) -> int:
    """
    Tool that multiplies two numbers.
    Args:
        x (int): The first number.
        y (int): The second number.
    Returns:
        int: The product of the two numbers.
    """
    return x * y

@mcp.tool()
def divide(x: int, y: int) -> float:
    """
    Tool that divides the first number by the second.
    Args:
        x (int): The numerator.
        y (int): The denominator.
    Returns:
        float: The result of the division.
    Raises:
        ValueError: If the denominator is zero.
    """
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y

if __name__ == "__main__":
    mcp.run(transport="stdio")