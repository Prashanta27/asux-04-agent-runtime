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
    print("=== AsuX AI Commerce Assistant Demo ===\n")

    # 1. Initialize Tool Runtime with Builtin Tools
    tool_registry = InMemoryToolRegistry()
    register_builtin_tools(tool_registry)
    tool_executor = ToolExecutor(registry=tool_registry)

    # 2. Initialize Prompt Engine with Project 2's storage
    prompt_registry = FilePromptRegistry(
        storage_dir="../02-prompt-engine/src/asux_prompt_engine/prompts"
    )
    prompt_engine = PromptEngine(registry=prompt_registry)

    # 3. Setup LLMClient with simulated responses for demonstration
    mock_llm_client = AsyncMock(spec=LLMClient)

    # Step 1: Model calls product_search tool
    # Step 2: Model gives final answer based on tool output
    res1 = GatewayResponse(
        request_id="req-1",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "product_search",
                    "arguments": {"query": "mechanical keyboard"},
                }
            )
        ),
    )
    res2 = GatewayResponse(
        request_id="req-2",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "final",
                    "message": "Found 1 product: 'mechanical keyboard' for $99.99.",
                }
            )
        ),
    )
    mock_llm_client.complete.side_effect = [res1, res2]

    # 4. Create Commerce Assistant
    assistant = CommerceAssistant(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=AgentConfig(max_steps=5),
    )

    # 5. Execute Request with Trusted Permission Context
    context = PermissionContext(
        principal_id="user_john",
        principal_type="customer",
        tenant_id="tenant_001",
        permissions=["product:read", "order:read"],
    )

    request = AgentRequest(
        goal="Find products matching mechanical keyboard",
        context=context,
        conversation_id="conv_101",
    )

    print(f"Goal: {request.goal}")
    print("Executing agent loop...")
    response = await assistant.run(request)

    print(f"\nFinal Status: {response.status}")
    print(f"Steps Executed: {response.steps_executed}")
    print(f"Tool Calls Executed: {response.tool_calls_executed}")
    print(f"Final Response: {response.final_response}")


if __name__ == "__main__":
    asyncio.run(main())
