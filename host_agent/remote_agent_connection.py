from collections.abc import Callable

import httpx
from langfuse import observe,get_client

from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    SendMessageRequest,
    SendMessageResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)

from dotenv import load_dotenv

langfuse = get_client()

load_dotenv()

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard, agent_url: str):
        print(f"agent_card: {agent_card}")
        print(f"agent_url: {agent_url}")
        # Increased timeout to 120 seconds for longer-running agent operations
        self._httpx_client = httpx.AsyncClient(timeout=120.0)
        self.agent_client = A2AClient(self._httpx_client, agent_card, url=agent_url)
        self.card = agent_card

    def get_agent(self) -> AgentCard:
        return self.card
    
    @observe(name="travel_planner_agent",as_type="agent")
    async def send_message(
        self, message_request: SendMessageRequest
    ) -> SendMessageResponse:
        # with langfuse.start_as_current_span(name="call_llm") as span:
        #     trace = langfuse.get_current_trace_id()

        # print(f"trace id in remote agent connection: {trace}")
        return await self.agent_client.send_message(message_request)
