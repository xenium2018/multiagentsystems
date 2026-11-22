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

    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    )
    weather_agent = create_weather_agent(model)
    travel_agent = create_travel_agent(model)
    conversation_history = []
    coordinator = create_coordinator_agent(
        model,
        weather_agent=weather_agent,
        travel_agent=travel_agent,
        conversation_history=conversation_history,
    )

    while True:
        question = input(
            "What would you like to know about your trip? (type 'exit' to quit) "
        ).strip()
        if not question:
            print("Please enter a question.")
            continue
        if question.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        conversation_history.append({"role": "user", "content": question})
        result = coordinator.invoke({"messages": conversation_history})
        assistant_reply = result["messages"][-1].content
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        print(assistant_reply)


if __name__ == "__main__":
    main()