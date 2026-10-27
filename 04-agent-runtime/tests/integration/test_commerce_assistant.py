import json
from unittest.mock import AsyncMock

import pytest
from asux_llm_gateway.models.messages import Message
from asux_llm_gateway.models.responses import GatewayResponse
from asux_prompt_engine.services.prompt_engine import PromptEngine
from asux_tool_runtime.executor.tool_executor import ToolExecutor
from asux_tool_runtime.models.context import PermissionContext

from asux_agent_runtime.config import AgentConfig
from asux_agent_runtime.models.request import AgentRequest
from asux_agent_runtime.models.status import AgentStatus
from asux_agent_runtime.runtime.agent import CommerceAssistant


@pytest.mark.asyncio
async def test_scenario_a_product_information(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    default_config: AgentConfig,
    full_permission_context: PermissionContext,
) -> None:
    """Scenario A: User asks to find products matching requirements."""
    res1 = GatewayResponse(
        request_id="req-prod-1",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "product_search",
                    "arguments": {"query": "ergonomic chair"},
                }
            )
        ),
    )
    res2 = GatewayResponse(
        request_id="req-prod-2",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "final",
                    "message": "We have 1 product matching 'ergonomic chair' priced at $99.99.",
                }
            )
        ),
    )
    mock_llm_client.complete.side_effect = [res1, res2]

    assistant = CommerceAssistant(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=default_config,
    )

    request = AgentRequest(
        goal="Find products matching ergonomic chair",
        context=full_permission_context,
    )
    response = await assistant.run(request)

    assert response.status == AgentStatus.COMPLETED
    assert response.tool_calls_executed == 1
    assert "ergonomic chair" in (response.final_response or "")


@pytest.mark.asyncio
async def test_scenario_b_order_lookup_clarification(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    default_config: AgentConfig,
    full_permission_context: PermissionContext,
) -> None:
    """Scenario B: User asks for order status without order ID -> agent asks clarification."""
    res1 = GatewayResponse(
        request_id="req-order-1",
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
    mock_llm_client.complete.return_value = res1

    assistant = CommerceAssistant(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=default_config,
    )

    request = AgentRequest(
        goal="Help me check the status of my order",
        context=full_permission_context,
    )
    response = await assistant.run(request)

    assert response.status == AgentStatus.NEEDS_INPUT
    assert "order number" in (response.final_response or "")
    assert response.tool_calls_executed == 0


@pytest.mark.asyncio
async def test_scenario_c_create_followup_task(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    default_config: AgentConfig,
    full_permission_context: PermissionContext,
) -> None:
    """Scenario C: User asks to create a follow-up task."""
    res1 = GatewayResponse(
        request_id="req-task-1",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "create_task",
                    "arguments": {
                        "title": "Follow up with customer",
                        "description": "Send catalog details",
                    },
                }
            )
        ),
    )
    res2 = GatewayResponse(
        request_id="req-task-2",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "final",
                    "message": "Follow-up task created successfully with ID task-12345.",
                }
            )
        ),
    )
    mock_llm_client.complete.side_effect = [res1, res2]

    assistant = CommerceAssistant(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=default_config,
    )

    request = AgentRequest(
        goal="Create a follow-up task for this customer",
        context=full_permission_context,
    )
    response = await assistant.run(request)

    assert response.status == AgentStatus.COMPLETED
    assert response.tool_calls_executed == 1
    assert "task-12345" in (response.final_response or "")


@pytest.mark.asyncio
async def test_scenario_d_tool_failure_recovery(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    default_config: AgentConfig,
    full_permission_context: PermissionContext,
) -> None:
    """Scenario D: Tool execution returns failure (e.g. unknown tool), agent recovers."""
    # Step 1: Model calls an invalid tool name
    res1 = GatewayResponse(
        request_id="req-fail-1",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "non_existent_tool",
                    "arguments": {"test": "val"},
                }
            )
        ),
    )
    # Step 2: Model recognizes failure and provides a safe explanation
    res2 = GatewayResponse(
        request_id="req-fail-2",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "final",
                    "message": "I could not perform this action because tool is not available.",
                }
            )
        ),
    )
    mock_llm_client.complete.side_effect = [res1, res2]

    assistant = CommerceAssistant(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=default_config,
    )

    request = AgentRequest(
        goal="Attempt unlisted tool",
        context=full_permission_context,
    )
    response = await assistant.run(request)

    assert response.status == AgentStatus.COMPLETED
    assert response.tool_calls_executed == 1
    assert response.tool_invocations[0].success is False


@pytest.mark.asyncio
async def test_scenario_e_permission_denied_boundary(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    default_config: AgentConfig,
    read_only_permission_context: PermissionContext,
) -> None:
    """Scenario E: Caller lacks permissions, Tool Runtime denies execution, handles securely."""
    # User is read_only (product:read), model tries to call create_task (requires task:write)
    res1 = GatewayResponse(
        request_id="req-perm-1",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "tool_call",
                    "tool_name": "create_task",
                    "arguments": {"title": "Unauthorized task"},
                }
            )
        ),
    )
    res2 = GatewayResponse(
        request_id="req-perm-2",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps(
                {
                    "action": "final",
                    "message": "You are not authorized to create tasks.",
                }
            )
        ),
    )
    mock_llm_client.complete.side_effect = [res1, res2]

    assistant = CommerceAssistant(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=default_config,
    )

    request = AgentRequest(
        goal="Create task without permission",
        context=read_only_permission_context,
    )
    response = await assistant.run(request)

    assert response.status == AgentStatus.COMPLETED
    assert response.tool_calls_executed == 1
    assert response.tool_invocations[0].success is False
    assert "permission" in (response.tool_invocations[0].error or "").lower()
