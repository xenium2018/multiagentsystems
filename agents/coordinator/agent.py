from langchain.agents import create_agent
from langchain.tools import tool


def _ask_specialist(agent, query: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return str(result["messages"][-1].content)


def create_coordinator_agent(model, weather_agent, travel_agent):
    @tool
    def ask_weather_specialist(query: str) -> str:
        """Ask the weather specialist about forecasts for a destination."""
        return _ask_specialist(weather_agent, query)

    @tool
    def ask_travel_specialist(query: str) -> str:
        """Ask the travel guide specialist about a destination."""
        return _ask_specialist(travel_agent, query)

    return create_agent(
        model=model,
        tools=[ask_weather_specialist, ask_travel_specialist],
        system_prompt=(
            "You are a travel assistant coordinating specialist agents. Delegate "
            "weather questions to the weather specialist and destination/travel "
            "questions to the travel specialist. For questions involving both, "
            "consult both specialists. Base factual answers on their results."
        ),
    )
