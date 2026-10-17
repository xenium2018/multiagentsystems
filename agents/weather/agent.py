from langchain.agents import create_agent
from langchain.tools import tool

from tools.weather import get_weather_forecast


@tool
def weather_forecast_tool(place: str, region: str = "") -> str:
    """Get a seven-day weather forecast for a place, optionally in a region."""
    return get_weather_forecast(place, region)


def create_weather_agent(model):
    return create_agent(
        model=model,
        tools=[weather_forecast_tool],
        system_prompt="You are a weather specialist. Answer using the weather forecast tool.",
    )
