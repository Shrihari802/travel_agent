# main.py
import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

# ADK imports
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.in_memory_session_service import InMemorySessionService

from agent import create_travel_planner_agent  # <-- Your factory function
from agent_executor import TravelPlannerAgentExecutor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: suppress noisy logs
logging.getLogger("google.adk").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


if __name__ == "__main__":
    # === 1. Create your ADK Agent (LlmAgent) ===
    travel_agent = create_travel_planner_agent()

    # === 2. Define Skill ===
    skill = AgentSkill(
        id="travel_planner",
        name="Travel Planner",
        description="Expert travel assistant for personalized itineraries, destination tips, and trip planning.",
        tags=["travel", "vacation", "itinerary", "tourism"],
        examples=[
            "Plan a 7-day trip to Japan in spring",
            "Best things to do in Italy for 10 days?",
            "Help me plan a budget trip to Thailand",
            "What’s the weather like in Bali in December?",
        ],
    )

    # === 3. Agent Card ===
    agent_card = AgentCard(
        name="Travel Planner Agent",
        description="Your personal AI travel planner — creates beautiful, realistic, and detailed trip itineraries tailored to you.",
        url="http://localhost:10001/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    # === 4. Create Runner — THIS IS THE FIX! ===
    runner = Runner(
        app_name=agent_card.name,           # Required
        agent=travel_agent,                 # Your LlmAgent instance ← CRITICAL
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),       # Recommended
        artifact_service=InMemoryArtifactService(),   # Recommended
    )

    # === 5. Executor ===
    travel_executor = TravelPlannerAgentExecutor(runner=runner, card=agent_card)

    # === 6. Request Handler ===
    request_handler = DefaultRequestHandler(
        agent_executor=travel_executor,
        task_store=InMemoryTaskStore(),
    )

    # === 7. A2A App ===
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    logger.info("Travel Planner Agent starting at http://localhost:10001")
    logger.info("   • Model: gpt-4o-mini (or your config)")
    logger.info("   • Streaming: Enabled")
    logger.info("   • Multi-turn chats: Supported")
    logger.info("   • Ready for tools (weather, flights, etc.)")

    uvicorn.run(
        app.build(),
        host="0.0.0.0",
        port=10001,
        log_level="info",
    )