# agent_executor.py
import logging
from typing import TYPE_CHECKING, List

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCard,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from a2a.utils import new_text_artifact

from google.adk import Runner
from google.genai import types

if TYPE_CHECKING:
    from google.adk.sessions.session import Session

from agent import create_travel_planner_agent  # Import the factory function


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Constants
DEFAULT_USER_ID = "self"


class TravelPlannerAgentExecutor(AgentExecutor):
    """
    AgentExecutor for the Travel Planner ADK Agent.
    Uses Google ADK Runner to execute the LlmAgent with full streaming, session persistence, and tool support.
    """

    def __init__(self, runner: Runner, card: AgentCard):
        self.runner = runner
        self._card = card
        self._active_sessions: set[str] = set()

        # Create the ADK agent once and reuse it
        self.adk_agent = create_travel_planner_agent()

    async def _process_request(
        self,
        new_message: types.Content,
        session_id: str,
        task_updater: TaskUpdater,
    ) -> None:
        """
        Core execution loop: runs the ADK agent in streaming mode and forwards events.
        """
        session_obj = await self._upsert_session(session_id)
        session_id = session_obj.id  # Use canonical session ID

        self._active_sessions.add(session_id)

        try:
            async for event in self.runner.run_async(
                session_id=session_id,
                user_id=DEFAULT_USER_ID,
                new_message=new_message,
            ):
                # Final response from agent (including tool results, final answer, etc.)
                if event.is_final_response():
                    parts = [
                        self._convert_genai_part_to_a2a(part)
                        for part in event.content.parts
                        if part.text or part.file_data or part.inline_data
                    ]
                    if parts:
                        await task_updater.add_artifact(parts)
                    await task_updater.update_status(TaskState.completed, final=True)
                    break

                # Streaming text updates (partial responses)
                if not event.get_function_calls():
                    text_parts = [
                        part.text
                        for part in event.content.parts
                        if part.text
                    ]
                    if text_parts:
                        combined_text = "".join(text_parts)
                        await task_updater.update_status(
                            TaskState.working,
                            message=task_updater.new_agent_message(
                                [TextPart(text=combined_text)]
                            ),
                        )
                    else:
                        # Fallback: use raw artifact streaming if no clean text
                        await task_updater.add_artifact(
                            [new_text_artifact(name="stream", text=".")]
                        )

                # Tool calls in progress → just keep working state
                else:
                    await task_updater.update_status(TaskState.working)

        except Exception as e:
            logger.exception("Error during travel planner execution")
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(
                    [TextPart(text=f"Sorry, something went wrong: {str(e)}")]
                ),
                final=True,
            )
        finally:
            self._active_sessions.discard(session_id)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # Initial task state
        if not context.current_task:
            await updater.update_status(TaskState.submitted)
        await updater.update_status(TaskState.working)

        # Convert incoming A2A message parts → Google GenAI format
        user_content = types.UserContent(
            parts=[
                self._convert_a2a_part_to_genai(part)
                for part in context.message.parts
            ]
        )

        await self._process_request(
            new_message=user_content,
            session_id=context.context_id,
            task_updater=updater,
        )

        logger.debug("[TravelPlanner] execute completed")

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Attempt to cancel ongoing travel planning session."""
        session_id = context.context_id

        if session_id in self._active_sessions:
            logger.info(f"Cancellation requested for active travel session: {session_id}")
            # Note: ADK currently doesn't support mid-stream cancellation,
            # but we clean up tracking and raise error as per convention
            self._active_sessions.discard(session_id)

        raise ServerError(error=UnsupportedOperationError("Cancellation not supported yet"))

    async def _upsert_session(self, session_id: str) -> "Session":
        """Get or create a persistent session."""
        session = await self.runner.session_service.get_session(
            app_name=self.runner.app_name,
            user_id=DEFAULT_USER_ID,
            session_id=session_id,
        )
        if session is None:
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name,
                user_id=DEFAULT_USER_ID,
                session_id=session_id,
            )
        return session

    # ———————— Conversion Helpers ————————

    def _convert_a2a_part_to_genai(self, part: Part) -> types.Part:
        part = part.root
        if isinstance(part, TextPart):
            return types.Part(text=part.text)
        if isinstance(part, FilePart):
            if isinstance(part.file, FileWithUri):
                return types.Part(
                    file_data=types.FileData(
                        file_uri=part.file.uri,
                        mime_type=part.file.mime_type,
                    )
                )
            if isinstance(part.file, FileWithBytes):
                return types.Part(
                    inline_data=types.Blob(
                        data=part.file.bytes,
                        mime_type=part.file.mime_type,
                    )
                )
        raise ValueError(f"Unsupported part type: {type(part)}")

    def _convert_genai_part_to_a2a(self, part: types.Part) -> Part:
        if part.text:
            return TextPart(text=part.text)
        if part.file_data:
            return FilePart(
                file=FileWithUri(
                    uri=part.file_data.file_uri,
                    mime_type=part.file_data.mime_type,
                )
            )
        if part.inline_data:
            return FilePart(
                file=FileWithBytes(
                    bytes=part.inline_data.data,
                    mime_type=part.inline_data.mime_type,
                )
            )
        raise ValueError(f"Unsupported GenAI part: {part}")