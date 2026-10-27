import json
from unittest.mock import AsyncMock

import pytest
from asux_llm_gateway.models.messages import Message
from asux_llm_gateway.models.responses import GatewayResponse
from asux_prompt_engine.services.prompt_engine import PromptEngine
from asux_tool_runtime.executor.tool_executor import ToolExecutor
from asux_tool_runtime.models.context import PermissionContext

from asux_agent_runtime.config import AgentConfig
from asux_agent_runtime.loop.agent_loop import AgentLoop
from asux_agent_runtime.models.request import AgentRequest
from asux_agent_runtime.models.status import AgentStatus


@pytest.mark.asyncio
async def test_max_steps_limit_enforced(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    full_permission_context: PermissionContext,
) -> None:
    # Model endlessly loops requesting tool calls
    tool_call_res = GatewayResponse(
        request_id="req-loop",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "product_search",
                    "arguments": {"query": "phone"},
                }
            )
        ),
    )
    mock_llm_client.complete.return_value = tool_call_res

    # Config with max_steps=3
    config = AgentConfig(max_steps=3, max_tool_calls=10, duplicate_call_safeguard=False)
    loop = AgentLoop(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=config,
    )

    req = AgentRequest(goal="Search forever", context=full_permission_context)
    response = await loop.run(req)

    assert response.status == AgentStatus.MAX_STEPS_REACHED
    assert response.steps_executed == 3
    assert "maximum" in (response.termination_reason or "").lower()


@pytest.mark.asyncio
async def test_max_tool_calls_limit_enforced(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    full_permission_context: PermissionContext,
) -> None:
    tool_call_res = GatewayResponse(
        request_id="req-loop",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "product_search",
                    "arguments": {"query": "laptop"},
                }
            )
        ),
    )
    mock_llm_client.complete.return_value = tool_call_res

    # max_tool_calls=2, max_steps=5
    config = AgentConfig(max_steps=5, max_tool_calls=2, duplicate_call_safeguard=False)
    loop = AgentLoop(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=config,
    )

    req = AgentRequest(goal="Search products", context=full_permission_context)
    response = await loop.run(req)

    assert response.status == AgentStatus.MAX_STEPS_REACHED
    assert response.tool_calls_executed == 2
    assert "maximum allowed tool calls" in (response.termination_reason or "").lower()


@pytest.mark.asyncio
async def test_consecutive_parsing_failure_limit(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    full_permission_context: PermissionContext,
) -> None:
    # Model consistently returns malformed, unparseable output
    invalid_res = GatewayResponse(
        request_id="req-invalid",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(content="I am not returning JSON today."),
    )
    mock_llm_client.complete.return_value = invalid_res

    config = AgentConfig(max_steps=5, max_consecutive_failures=2)
    loop = AgentLoop(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=config,
    )

    req = AgentRequest(goal="Do something", context=full_permission_context)
    response = await loop.run(req)

    assert response.status == AgentStatus.FAILED
    assert "consecutive decision parsing failures" in (response.termination_reason or "").lower()


@pytest.mark.asyncio
async def test_duplicate_failed_tool_call_safeguard(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    read_only_permission_context: PermissionContext,
) -> None:
    # Model tries to call a tool it has no permission for (create_task) repeatedly
    unauthorized_call = GatewayResponse(
        request_id="req-unauth",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "create_task",
                    "arguments": {"title": "Restock products"},
                }
            )
        ),
    )
    mock_llm_client.complete.return_value = unauthorized_call

    config = AgentConfig(
        max_steps=5,
        max_consecutive_failures=2,
        duplicate_call_safeguard=True,
    )
    loop = AgentLoop(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=config,
    )

    req = AgentRequest(goal="Create task without permission", context=read_only_permission_context)
    response = await loop.run(req)

    assert response.status == AgentStatus.FAILED
    assert "duplicate consecutive failed call" in (response.termination_reason or "").lower()


@pytest.mark.asyncio
async def test_request_max_steps_cannot_override_global_config(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    full_permission_context: PermissionContext,
) -> None:
    # Global config max_steps is 3, but request attempts to request 100
    tool_call_res = GatewayResponse(
        request_id="req-loop",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "product_search",
                    "arguments": {"query": "item"},
                }
            )
        ),
    )
    mock_llm_client.complete.return_value = tool_call_res

    config = AgentConfig(max_steps=3, duplicate_call_safeguard=False)
    loop = AgentLoop(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=config,
    )

    req = AgentRequest(
        goal="Attempt override",
        context=full_permission_context,
        max_steps=100,
    )
    response = await loop.run(req)

    # Must stop at config's 3, NOT 100!
    assert response.steps_executed == 3
    assert response.status == AgentStatus.MAX_STEPS_REACHED
