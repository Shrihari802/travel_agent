# pylint: disable=logging-fstring-interpolation
import asyncio
import json
import os
import uuid

from typing import Any

import httpx
from google.adk.models.lite_llm import LiteLlm
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from remote_agent_connection import (
    RemoteAgentConnections,
    TaskUpdateCallback,
)
from langfuse import get_client, observe
import time
import requests

load_dotenv()
langfuse = get_client()
def convert_part(part: Part, tool_context: ToolContext):
    """Convert a part to text. Only text parts are supported."""
    if part.type == "text":
        return part.text

    return f"Unknown type: {part.type}"


def convert_parts(parts: list[Part], tool_context: ToolContext):
    """Convert parts to text."""
    rval = []
    for p in parts:
        rval.append(convert_part(p, tool_context))
    return rval


def create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> dict[str, Any]:
    """Helper function to create the payload for sending a task."""
    payload: dict[str, Any] = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": text}],
            "messageId": uuid.uuid4().hex,
        },
    }

    if task_id:
        payload["message"]["taskId"] = task_id

    if context_id:
        payload["message"]["contextId"] = context_id
    return payload


class RoutingAgent:
    """The Routing agent.

    This is the agent responsible for choosing which remote seller agents to send
    tasks to and coordinate their work.
    """

    def __init__(
        self,
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""

    async def _async_init_components(self, remote_agent_addresses: list[str]) -> None:
        """Asynchronous part of initialization."""
        # Use a single httpx.AsyncClient for all card resolutions for efficiency
        async with httpx.AsyncClient(timeout=60.0) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)  # Constructor is sync
                try:
                    card = (
                        await card_resolver.get_agent_card()
                    )  # get_agent_card is async

                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e:  # Catch other potential errors
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")

        # Populate self.agents using the logic from original __init__ (via list_remote_agents)
        agent_info = []
        for agent_detail_dict in self.list_remote_agents():
            agent_info.append(json.dumps(agent_detail_dict))
        self.agents = "\n".join(agent_info)

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: list[str],
        task_callback: TaskUpdateCallback | None = None,
    ) -> "RoutingAgent":
        """Create and asynchronously initialize an instance of the RoutingAgent."""
        instance = cls(task_callback)
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        """Create an instance of the RoutingAgent."""
        LITELLM_MODEL = "azure/gpt-4o-mini"
        return Agent(
            model=LiteLlm(
                model=LITELLM_MODEL,
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                # api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2024-05-01-preview",
                api_base=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            ),
            name="Routing_agent",
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                "This Routing agent orchestrates the travel planning process by delegating tasks to specialized remote agents including travel planning and budget estimation. "
            ),
            tools=[
                self.send_message,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        """Generate the root instruction for the RoutingAgent."""
        current_agent = self.check_active_agent(context)
        return f"""
        **Role:** Host agent for the agent-to-agent protocol; delegates queries to specialized remote agents with maximum efficiency.

**Core Directives:**

* **Task Delegation:** Use the `send_message` function to assign precise, actionable tasks to remote agents.
* **Full Context Provision:** If an agent repeatedly asks for user confirmation, it likely lacks conversation history. Include all relevant context in the task to prevent this.
* **Autonomous Multi-Agent Engagement:** Engage any required agents directly—never seek user permission or preference. If multiple agents are needed, orchestrate them seamlessly.
* **Intelligent Inter-Agent Collaboration:** Instruct agents to determine if they need data from another agent. **If Agent A says "I need X to proceed" and Agent B can provide X, immediately query Agent B, then resubmit the updated task to Agent A.**
* **Transparent Output:** Deliver the full, unedited response from the final agent(s) to the user.
* **Confirmation Handling:** Only relay confirmation requests to the user if not already provided. Never confirm on behalf of the user.
* **Minimal Context Sharing:** Share only task-relevant context with each agent. Omit irrelevant details.
* **No Redundancy:** Never ask agents to confirm actions or information.
* **Tool-Only Responses:** Rely exclusively on tools and agents. If data is missing, request clarification from the user—never assume.
* **Recency Priority:** Base decisions on the most recent user message.
* **Active Agent Routing:** Route follow-up queries to the currently active agent using task updates.

**Efficiency Rule:** 
> **Dependency Resolution Loop:**  
> If Agent A blocks on missing info → Identify Agent B that can supply it → Query B → Feed result back to A → Repeat until A completes or escalates.
        **Agent Roster:**

        * Available Agents: `{self.agents}`
        * Currently Active Seller Agent: `{current_agent['active_agent']}`
                """

    def check_active_agent(self, context: ReadonlyContext):
        state = context.state
        if (
            "session_id" in state
            and "session_active" in state
            and state["session_active"]
            and "active_agent" in state
        ):
            return {"active_agent": f'{state["active_agent"]}'}
        return {"active_agent": "None"}

    def before_model_callback(self, callback_context: CallbackContext, llm_request):
        state = callback_context.state
        if "session_active" not in state or not state["session_active"]:
            if "session_id" not in state:
                state["session_id"] = str(uuid.uuid4())
            state["session_active"] = True

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.cards:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            print(f"Found agent card: {card.model_dump(exclude_none=True)}")
            print("=" * 100)
            remote_agent_info.append(
                {"name": card.name, "description": card.description}
            )
        return remote_agent_info

    @observe(name = "travel_planner_agent",as_type= "agent")
    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext):
        """Sends a task to remote seller agent.

        This will send a message to the remote agent named agent_name.

        Args:
            agent_name: The name of the agent to send the task to.
            task: The comprehensive conversation context summary
                and goal to be achieved regarding user inquiry and purchase request.
            tool_context: The tool context this method runs in.

        Yields:
            A dictionary of JSON data.
        """
        start_time = time.time()
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        state = tool_context.state
        state["active_agent"] = agent_name
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f"Client not available for {agent_name}")
        # task_id = state["task_id"] if "task_id" in state else str(uuid.uuid4())
        task_id = state.get("task_id")
        if task_id and agent_name == state.get("active_agent"):
            params_data["message"]["taskId"] = task_id

        if "context_id" in state:
            context_id = state["context_id"]
        else:
            context_id = str(uuid.uuid4())

        message_id = ""
        metadata = {}
        if "input_message_metadata" in state:
            metadata.update(**state["input_message_metadata"])
            if "message_id" in state["input_message_metadata"]:
                message_id = state["input_message_metadata"]["message_id"]
        if not message_id:
            message_id = str(uuid.uuid4())
        # Build only the *params* part for MessageSendParams
        params_data = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": message_id,
            }
        }

        if task_id:
            params_data["message"]["taskId"] = task_id

        if context_id:
            params_data["message"]["contextId"] = context_id

        # Validate only the params part
        message_send_params = MessageSendParams.model_validate(params_data)

        # Now build the full JSON-RPC request
        message_request = SendMessageRequest(
            id=message_id,
            params=message_send_params
        )
        print(message_request.model_dump_json(exclude_none=True, indent=2))
        try:
            send_response: SendMessageResponse = await client.send_message(
                message_request=message_request
            )
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                print(f"ERROR: Timeout calling agent {agent_name}: {e}")
                return {
                    "status": "error",
                    "message": f"Request to {agent_name} timed out. Please ensure the agent is running and try again.",
                    "agent_name": agent_name,
                }
            else:
                print(f"ERROR: Failed to call agent {agent_name}: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to communicate with {agent_name}: {error_msg}",
                    "agent_name": agent_name,
                }
        
        trace_id = langfuse.get_current_trace_id()
        
        # Determine agent name for tracking
        agent_tracking_name = agent_name.lower().replace(" ", "_").replace("&", "_")
        payload = {
                "agent_name": agent_tracking_name,
                "project_name": "MakeEasyTravel",
                "trace_id": trace_id,
                "processed_status": False
        }
        try:
            requests.post("https://aks.agentix-xebia.com/agent-control-tower/api/observe/trace_id", json=payload, timeout=10, verify=False)
        except requests.RequestException as exc:
            print(f"Warning: failed to push trace metadata: {exc}")
 
             
        
        print(f"{trace_id}send_response",
            send_response.model_dump_json(exclude_none=True, indent=2)
        )

        if not isinstance(send_response.root, SendMessageSuccessResponse):
            print("received non-success response. Aborting get task ")
            return {
                "status": "error",
                "message": f"{agent_name} returned a non-success response",
                "agent_name": agent_name,
            }

        if not isinstance(send_response.root.result, Task):
            print("received non-task response. Aborting get task ")
            return {
                "status": "error",
                "message": f"{agent_name} returned a non-task response",
                "agent_name": agent_name,
            }

        task_result = send_response.root.result.model_dump(exclude_none=True)

        return {
            "status": "success",
            "agent_name": agent_name,
            "elapsed_seconds": round(time.time() - start_time, 2),
            "task": task_result,
        }


def _get_initialized_routing_agent_sync() -> Agent:
    """Synchronously creates and initializes the RoutingAgent."""

    async def _async_main() -> Agent:
        routing_agent_instance = await RoutingAgent.create(
            remote_agent_addresses=[
                # os.getenv("MANAGER_AGENT_URL", "http://localhost:10001"),
                # os.getenv("DEPTPARTMENT_AGENT_URL", "http://localhost:10002"),
                os.getenv("TRAVEL_AGENT_URL", "http://0.0.0.0:10001"),
                os.getenv("BUDGET_AGENT_URL", "http://0.0.0.0:10007"),
                
                # os.getenv("CAPITAL_AGENT_URL", "https://func-a2a5055-a6fufrexfuaed0f7.eastus2-01.azurewebsites.net/api/chat"),
                os.getenv("TRANSALTOR_AGENT_URL", "https://5i3kzb5qsydzvuywzqg62cqqzi0pvjkz.lambda-url.us-east-2.on.aws/api/chat"),
            ]
        )
        return routing_agent_instance.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(
                f"Warning: Could not initialize RoutingAgent with asyncio.run(): {e}. "
                "This can happen if an event loop is already running (e.g., in Jupyter). "
                "Consider initializing RoutingAgent within an async function in your application."
            )
        raise


root_agent = _get_initialized_routing_agent_sync()