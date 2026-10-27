import json
from unittest.mock import AsyncMock

import pytest
from asux_llm_gateway.exceptions.errors import GatewayError, GatewayTimeoutError
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
async def test_gateway_timeout_recovery_and_success(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    default_config: AgentConfig,
    full_permission_context: PermissionContext,
) -> None:
    # First call times out, second call succeeds
    res_success = GatewayResponse(
        request_id="req-recovered",
        model="gpt-4o-mini",
        provider="mock",
        message=Message.assistant(
            content=json.dumps({"action": "final", "message": "Recovered from timeout."})
        ),
    )
    mock_llm_client.complete.side_effect = [
        GatewayTimeoutError("Upstream provider timed out"),
        res_success,
    ]

    loop = AgentLoop(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=default_config,
    )

    req = AgentRequest(goal="Test timeout recovery", context=full_permission_context)
    response = await loop.run(req)

    assert response.status == AgentStatus.COMPLETED
    assert response.final_response == "Recovered from timeout."


@pytest.mark.asyncio
async def test_gateway_persistent_errors_abort(
    mock_llm_client: AsyncMock,
    prompt_engine: PromptEngine,
    tool_executor: ToolExecutor,
    full_permission_context: PermissionContext,
) -> None:
    # Consecutive gateway errors exceed limit
    mock_llm_client.complete.side_effect = GatewayError("Provider API 500 error")

    config = AgentConfig(max_consecutive_failures=2)
    loop = AgentLoop(
        llm_client=mock_llm_client,
        prompt_engine=prompt_engine,
        tool_executor=tool_executor,
        config=config,
    )

    req = AgentRequest(goal="Test gateway failure", context=full_permission_context)
    response = await loop.run(req)

    assert response.status == AgentStatus.FAILED
    assert "repeated llm errors" in (response.termination_reason or "").lower()
