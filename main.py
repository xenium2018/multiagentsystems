import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.coordinator.agent import create_coordinator_agent
from agents.travel.agent import create_travel_agent
from agents.weather.agent import create_weather_agent


def main() -> None:
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to a .env file in the project folder."
        )

    question = input("What would you like to know about your trip? ").strip()
    if not question:
        raise ValueError("Please enter a question.")

    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    )
    weather_agent = create_weather_agent(model)
    travel_agent = create_travel_agent(model)
    coordinator = create_coordinator_agent(
        model,
        weather_agent=weather_agent,
        travel_agent=travel_agent,
    )

    result = coordinator.invoke({"messages": [{"role": "user", "content": question}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()