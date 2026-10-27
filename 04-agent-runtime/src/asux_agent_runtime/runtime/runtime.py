from collections.abc import AsyncGenerator

from asux_llm_gateway.clients.llm_client import LLMClient
from asux_prompt_engine.services.prompt_engine import PromptEngine
from asux_tool_runtime.executor.tool_executor import ToolExecutor

from asux_agent_runtime.config import AgentConfig
from asux_agent_runtime.models.events import AgentEvent
from asux_agent_runtime.models.request import AgentRequest
from asux_agent_runtime.models.response import AgentResponse
from asux_agent_runtime.runtime.agent import CommerceAssistant


class AgentRuntime:
    """
    Unified manager and factory for AsuX AI Agents.
    Maintains shared references to gateway, prompt engine, and tool runtime.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_engine: PromptEngine,
        tool_executor: ToolExecutor,
        default_config: AgentConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_engine = prompt_engine
        self.tool_executor = tool_executor
        self.default_config = default_config or AgentConfig()

    def create_commerce_assistant(self, config: AgentConfig | None = None) -> CommerceAssistant:
        """
        Creates an instance of the Single AI Commerce Assistant.
        """
        effective_config = config or self.default_config
        return CommerceAssistant(
            llm_client=self.llm_client,
            prompt_engine=self.prompt_engine,
            tool_executor=self.tool_executor,
            config=effective_config,
        )

    async def execute(
        self, request: AgentRequest, config: AgentConfig | None = None
    ) -> AgentResponse:
        """
        Directly executes an agent request using a CommerceAssistant.
        """
        assistant = self.create_commerce_assistant(config)
        return await assistant.run(request)

    async def execute_stream(
        self, request: AgentRequest, config: AgentConfig | None = None
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Directly executes an agent request yielding progress events.
        """
        assistant = self.create_commerce_assistant(config)
        async for event in assistant.run_stream(request):
            yield event
