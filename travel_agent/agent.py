# agent.py
import os
from typing import Any

from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

load_dotenv()


def create_travel_planner_agent() -> LlmAgent:
    """
    Creates and returns a fully configured Travel Planner agent using Azure OpenAI via Google ADK + LiteLLM.
    """
    # Load Azure-specific environment variables
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")      # e.g. https://your-resource.openai.azure.com/
    AZURE_MODEL = os.getenv("AZURE_OPENAI_MODEL")            # e.g. gpt-4o-mini or your deployment name
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

    if not all([AZURE_ENDPOINT, AZURE_MODEL, AZURE_API_KEY]):
        raise ValueError(
            "Missing required Azure OpenAI environment variables. "
            "Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_MODEL, and AZURE_OPENAI_API_KEY in your .env file."
        )

    # Construct the Azure model identifier expected by LiteLLM
    # Format: azure/<your-deployment-name>
    # Note: You use the DEPLOYMENT NAME here, not the base model name
    MODEL_NAME = f"azure/{AZURE_MODEL}"

    return LlmAgent(
        model=LiteLlm(
            model=MODEL_NAME,
            api_key=AZURE_API_KEY,
            api_base=AZURE_ENDPOINT.rstrip("/") + "/",  # Ensure proper trailing slash
            api_version="2024-08-01-preview",           # Recommended stable version as of 2025
            temperature=0.7,
            max_tokens=4096,
        ),
        name="travel_planner_agent",
        description="Expert travel assistant that creates personalized, realistic trip plans and gives destination advice.",
        instruction="""
You are an expert travel assistant specializing in trip planning, destination recommendations, itineraries, and practical travel advice.

Core guidelines:
- Always be helpful, enthusiastic, and realistic.
- Take into account budget, seasonality, travel time, weather, local events, and visa requirements when relevant.
- Create day-by-day itineraries that are feasible (respect opening hours, travel distances, meal times, rest).
- Balance famous attractions with authentic, off-the-beaten-path experiences.
- Recommend local cuisine and cultural activities.
- Use clear formatting: headings, bullet points, numbered days, bold for emphasis.
- If information is uncertain or outdated, mention it and suggest the user verify.

When the user asks for an itinerary, structure your final answer like this (JSON envelope for easy parsing if needed, otherwise rich Markdown):

{
  "trip_summary": "Brief overview of the trip",
  "duration": "X days / Y nights",
  "itinerary": [
    {
      "day": 1,
      "date": "YYYY-MM-DD (if known)",
      "title": "Arrival & City Exploration",
      "schedule": [
        "Morning: ...",
        "Afternoon: ...",
        "Evening: Dinner at ..."
      ],
      "accommodation": "Hotel name or area",
      "notes": "Any tips or alternatives"
    }
    // ... more days
  ],
  "practical_info": {
    "best_time_to_visit": "...",
    "budget_estimate": "...",
    "transportation_tips": "...",
    "packing_suggestions": "..."
  }
}

You may enrich responses with tools in the future (weather, flights, etc.), but for now rely solely on your knowledge.
""",
        tools=[
            # Add tools later when ready
        ],
    )


# Convenience: instantiate directly when running the file
if __name__ == "__main__":
    agent = create_travel_planner_agent()
    print(f"Travel Planner Agent created: {agent.name}")
    print(f"Using Azure OpenAI model: {os.getenv('AZURE_OPENAI_MODEL')}")