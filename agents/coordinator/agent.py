from langchain.agents import create_agent
from langchain.tools import tool


def create_coordinator_agent(model, weather_agent, travel_agent):
    @tool
    def ask_weather_specialist(query: str) -> str:
        """Ask the weather specialist about forecasts for a destination."""
        result = weather_agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        return str(result["messages"][-1].content)

    @tool
    def ask_travel_specialist(query: str) -> str:
        """Ask the travel guide specialist about a destination."""
        result = travel_agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        return str(result["messages"][-1].content)

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
