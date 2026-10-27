import asyncio
import json
from unittest.mock import AsyncMock

from asux_llm_gateway.clients.llm_client import LLMClient
from asux_llm_gateway.models.messages import Message
from asux_llm_gateway.models.responses import GatewayResponse
from asux_prompt_engine.registry.file_registry import FilePromptRegistry
from asux_prompt_engine.services.prompt_engine import PromptEngine
from asux_tool_runtime.executor.tool_executor import ToolExecutor
from asux_tool_runtime.models.context import PermissionContext
from asux_tool_runtime.registry.tool_registry import InMemoryToolRegistry
from asux_tool_runtime.tools.catalog import register_builtin_tools

from asux_agent_runtime.config import AgentConfig
from asux_agent_runtime.models.request import AgentRequest
from asux_agent_runtime.runtime.agent import CommerceAssistant


async def main() -> None:
    print("=== AsuX AI Commerce Assistant Streaming Demo ===\n")

    tool_registry = InMemoryToolRegistry()
    register_builtin_tools(tool_registry)
    tool_executor = ToolExecutor(registry=tool_registry)

    prompt_registry = FilePromptRegistry(
        storage_dir="../02-prompt-engine/src/asux_prompt_engine/prompts"
    )
    prompt_engine = PromptEngine(registry=prompt_registry)

    mock_llm_client = AsyncMock(spec=LLMClient)
    res = GatewayResponse(
        request_id="req-stream-1",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "clarification",
                    "message": "Please provide your order number so I can check its status.",
                }
            )
        ),
    )
    mock_llm_client.complete.return_value = res

    assistant = CommerceAssistant(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=AgentConfig(),
    )

    context = PermissionContext(
        principal_id="user_alice",
        principal_type="customer",
        tenant_id="tenant_001",
        permissions=["order:read"],
    )

    request = AgentRequest(
        goal="Check my order status",
        context=context,
    )

    print(f"Goal: {request.goal}")
    print("Streaming events:")
    async for event in assistant.run_stream(request):
        print(f"[{event.event_type.value}] step={event.step} data={event.data}")


if __name__ == "__main__":
    asyncio.run(main())
