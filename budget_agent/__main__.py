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

from agent import create_budget_agent  # <-- Your factory function
from agent_executor import BudgetAgentExecutor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: suppress noisy logs
logging.getLogger("google.adk").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


if __name__ == "__main__":
    # === 1. Create your ADK Agent (LlmAgent) ===
    budget_agent = create_budget_agent()

    # === 2. Define Skill ===
    skill = AgentSkill(
        id="travel_budget",
        name="Travel Budget Agent",
        description="Expert travel budget agent for estimating trip costs, breaking down expenses, and optimizing travel plans based on budget limits.",
        tags=["travel", "budget", "cost", "expenses", "optimization"],
        examples=[
            "Estimate the cost for a 7-day trip to Japan",
            "Break down the budget for my Paris itinerary",
            "Optimize my travel plan to fit within $3000",
            "Calculate costs for flights, hotels, and activities",
        ],
    )

    # === 3. Agent Card ===
    agent_card = AgentCard(
        name="Travel Budget Agent",
        description="Your personal AI travel budget assistant — estimates trip costs, breaks down expenses, and optimizes travel plans based on budget limits.",
        url="http://localhost:10007/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    # === 4. Create Runner — THIS IS THE FIX! ===
    runner = Runner(
        app_name=agent_card.name,           # Required
        agent=budget_agent,                 # Your LlmAgent instance ← CRITICAL
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),       # Recommended
        artifact_service=InMemoryArtifactService(),   # Recommended
    )

    # === 5. Executor ===
    budget_executor = BudgetAgentExecutor(runner=runner, card=agent_card)

    # === 6. Request Handler ===
    request_handler = DefaultRequestHandler(
        agent_executor=budget_executor,
        task_store=InMemoryTaskStore(),
    )

    # === 7. A2A App ===
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    logger.info("Travel Budget Agent starting at http://localhost:10007")
    logger.info("   • Model: gpt-4o-mini (or your config)")
    logger.info("   • Streaming: Enabled")
    logger.info("   • Multi-turn chats: Supported")
    logger.info("   • Ready for tools (real-time price APIs, currency converters, etc.)")

    uvicorn.run(
        app.build(),
        host="0.0.0.0",
        port=10007,
        log_level="info",
    )
