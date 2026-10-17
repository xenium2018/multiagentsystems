from langchain.agents import create_agent
from langchain.tools import tool

from tools.travel import search_travel_guide


@tool
def travel_guide_tool(place: str) -> str:
    """Look up travel information for a destination in Wikivoyage."""
    return search_travel_guide(place)


def create_travel_agent(model):
    return create_agent(
        model=model,
        tools=[travel_guide_tool],
        system_prompt="You are a travel guide specialist. Answer using the travel guide tool.",
    )
