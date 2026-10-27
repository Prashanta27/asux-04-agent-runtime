from collections.abc import AsyncGenerator

from asux_llm_gateway.clients.llm_client import LLMClient
from asux_prompt_engine.services.prompt_engine import PromptEngine
from asux_tool_runtime.executor.tool_executor import ToolExecutor

from asux_agent_runtime.config import AgentConfig
from asux_agent_runtime.loop.agent_loop import AgentLoop
from asux_agent_runtime.models.events import AgentEvent
from asux_agent_runtime.models.request import AgentRequest
from asux_agent_runtime.models.response import AgentResponse


class CommerceAssistant:
    """
    Dedicated AI Commerce Assistant agent.
    Encapsulates the agent loop configured for commerce workflows.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_engine: PromptEngine,
        tool_executor: ToolExecutor,
        config: AgentConfig | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.loop = AgentLoop(
            llm_client=llm_client,
            prompt_engine=prompt_engine,
            tool_executor=tool_executor,
            config=self.config,
        )

    async def run(self, request: AgentRequest) -> AgentResponse:
        """
        Executes a single commerce goal to completion.
        """
        return await self.loop.run(request)

    async def run_stream(self, request: AgentRequest) -> AsyncGenerator[AgentEvent, None]:
        """
        Executes a commerce goal while streaming progress events.
        """
        async for event in self.loop.run_stream(request):
            yield event
