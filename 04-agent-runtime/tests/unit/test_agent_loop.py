import json
from unittest.mock import AsyncMock

import pytest
from asux_llm_gateway.models.messages import Message, MessageRole
from asux_llm_gateway.models.responses import GatewayResponse
from asux_prompt_engine.services.prompt_engine import PromptEngine
from asux_tool_runtime.executor.tool_executor import ToolExecutor
from asux_tool_runtime.models.context import PermissionContext

from asux_agent_runtime.config import AgentConfig
from asux_agent_runtime.loop.agent_loop import AgentLoop
from asux_agent_runtime.models.events import AgentEventType
from asux_agent_runtime.models.request import AgentRequest
from asux_agent_runtime.models.status import AgentStatus


@pytest.mark.asyncio
async def test_agent_loop_direct_final_response(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    default_config: AgentConfig,
    full_permission_context: PermissionContext,
) -> None:
    # Model decides final answer on first step
    mock_llm_client.complete.return_value = GatewayResponse(
        request_id="req-1",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps({"action": "final", "message": "Direct answer to user."})
        ),
    )

    loop = AgentLoop(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=default_config,
    )

    req = AgentRequest(
        goal="What is your return policy?",
        context=full_permission_context,
    )

    response = await loop.run(req)
    assert response.status == AgentStatus.COMPLETED
    assert response.final_response == "Direct answer to user."
    assert response.steps_executed == 1
    assert response.tool_calls_executed == 0


@pytest.mark.asyncio
async def test_agent_loop_clarification_response(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    default_config: AgentConfig,
    full_permission_context: PermissionContext,
) -> None:
    # Model decides clarification is required
    mock_llm_client.complete.return_value = GatewayResponse(
        request_id="req-1",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "clarification",
                    "message": "Which order number would you like me to inspect?",
                }
            )
        ),
    )

    loop = AgentLoop(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=default_config,
    )

    req = AgentRequest(
        goal="Check my order",
        context=full_permission_context,
    )

    response = await loop.run(req)
    assert response.status == AgentStatus.NEEDS_INPUT
    assert response.final_response == "Which order number would you like me to inspect?"
    assert response.steps_executed == 1
    assert response.tool_calls_executed == 0


@pytest.mark.asyncio
async def test_agent_loop_single_tool_call_and_message_ordering(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    default_config: AgentConfig,
    full_permission_context: PermissionContext,
) -> None:
    # Step 1: Tool call decision
    res1 = GatewayResponse(
        request_id="req-1",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "product_search",
                    "arguments": {"query": "wireless mouse"},
                }
            )
        ),
    )
    # Step 2: Final answer grounded on tool output
    res2 = GatewayResponse(
        request_id="req-2",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "final",
                    "message": "Found 1 wireless mouse priced at $99.99.",
                }
            )
        ),
    )
    mock_llm_client.complete.side_effect = [res1, res2]

    loop = AgentLoop(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=default_config,
    )

    req = AgentRequest(
        goal="Search for wireless mouse",
        context=full_permission_context,
    )

    response = await loop.run(req)
    assert response.status == AgentStatus.COMPLETED
    assert response.steps_executed == 2
    assert response.tool_calls_executed == 1
    assert len(response.tool_invocations) == 1
    assert response.tool_invocations[0].tool_name == "product_search"
    assert response.tool_invocations[0].success is True

    # Verify message sequence passed to the 2nd LLM call
    second_call_req = mock_llm_client.complete.call_args_list[1][0][0]
    messages = second_call_req.messages
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].role == MessageRole.USER
    assert messages[1].content == "Search for wireless mouse"
    assert messages[2].role == MessageRole.ASSISTANT
    assert "product_search" in (messages[2].content or "")
    assert messages[3].role == MessageRole.TOOL
    assert "Mock Product" in (messages[3].content or "")


@pytest.mark.asyncio
async def test_agent_loop_streaming_events(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    default_config: AgentConfig,
    full_permission_context: PermissionContext,
) -> None:
    res1 = GatewayResponse(
        request_id="req-1",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps({"action": "final", "message": "Streamed final response."})
        ),
    )
    mock_llm_client.complete.return_value = res1

    loop = AgentLoop(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=default_config,
    )

    req = AgentRequest(
        goal="Say hello",
        context=full_permission_context,
    )

    events = [event async for event in loop.run_stream(req)]
    event_types = [e.event_type for e in events]

    assert AgentEventType.RUN_STARTED in event_types
    assert AgentEventType.MODEL_CALL_STARTED in event_types
    assert AgentEventType.MODEL_DECISION_RECEIVED in event_types
    assert AgentEventType.FINAL_RESPONSE_AVAILABLE in event_types
    assert AgentEventType.RUN_TERMINATED in event_types
