from mcp.server.fastmcp import FastMCP
from typing import List
mcp= FastMCP("Math")

@mcp.tool()
def weather(location: str) -> str:
    """
    Tool that retrieves the current weather for a given location.
    Args:
        location (str): The location to get the weather for.
    Returns:
        str: A string describing the current weather conditions.
    """
    # In a real implementation, you would fetch weather data from an API.
    # Here, we return a mock response for demonstration purposes.
    mock_weather_data = {
        "New York": "Sunny, 25°C",
        "Los Angeles": "Cloudy, 22°C",
        "Chicago": "Rainy, 18°C",
        "Houston": "Thunderstorms, 30°C",
        "Miami": "Hot and Humid, 32°C",
    }
    return mock_weather_data.get(location, "Weather data not available for this location.")

if __name__ == "__main__":
    mcp.run(transport="sse")