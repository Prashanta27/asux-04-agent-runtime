import pytest
from asux_tool_runtime.models.context import PermissionContext
from pydantic import ValidationError

from asux_agent_runtime.config import AgentConfig
from asux_agent_runtime.exceptions.errors import AgentConfigurationError
from asux_agent_runtime.models.decision import AgentAction, AgentDecision
from asux_agent_runtime.models.request import AgentRequest
from asux_agent_runtime.models.response import AgentResponse, ToolInvocationRecord
from asux_agent_runtime.models.status import AgentStatus


def test_agent_status_values() -> None:
    assert AgentStatus.PENDING == "pending"
    assert AgentStatus.RUNNING == "running"
    assert AgentStatus.COMPLETED == "completed"
    assert AgentStatus.NEEDS_INPUT == "needs_input"
    assert AgentStatus.MAX_STEPS_REACHED == "max_steps_reached"
    assert AgentStatus.FAILED == "failed"
    assert AgentStatus.CANCELLED == "cancelled"


def test_agent_request_validation(read_only_permission_context: PermissionContext) -> None:
    req = AgentRequest(
        goal="Find laptop",
        context=read_only_permission_context,
        conversation_id="conv_1",
    )
    assert req.goal == "Find laptop"
    assert req.context.principal_id == "user_guest"

    # Empty goal must raise ValidationError
    with pytest.raises(ValidationError):
        AgentRequest(goal="", context=read_only_permission_context)


def test_agent_decision_validation() -> None:
    # Final action
    d1 = AgentDecision(action=AgentAction.FINAL, message="Done")
    assert d1.action == AgentAction.FINAL
    assert d1.message == "Done"

    # Clarification action
    d2 = AgentDecision(action=AgentAction.CLARIFICATION, message="Which brand?")
    assert d2.action == AgentAction.CLARIFICATION

    # Tool call action
    d3 = AgentDecision(
        action=AgentAction.TOOL_CALL,
        tool_name="product_search",
        arguments={"query": "laptop"},
    )
    assert d3.tool_name == "product_search"
    assert d3.arguments == {"query": "laptop"}

    # Invalid tool call: missing tool_name
    with pytest.raises(ValidationError):
        AgentDecision(action=AgentAction.TOOL_CALL, arguments={})

    # Invalid tool call: missing arguments
    with pytest.raises(ValidationError):
        AgentDecision(action=AgentAction.TOOL_CALL, tool_name="product_search")

    # Invalid final: missing message
    with pytest.raises(ValidationError):
        AgentDecision(action=AgentAction.FINAL)


def test_agent_response_model() -> None:
    inv = ToolInvocationRecord(
        tool_name="product_search",
        arguments={"query": "phone"},
        success=True,
        duration_ms=12.5,
    )
    res = AgentResponse(
        run_id="run_123",
        status=AgentStatus.COMPLETED,
        final_response="Found phones",
        steps_executed=2,
        tool_calls_executed=1,
        tool_invocations=[inv],
    )
    assert res.run_id == "run_123"
    assert res.status == AgentStatus.COMPLETED
    assert len(res.tool_invocations) == 1


def test_agent_config_validation() -> None:
    cfg = AgentConfig(max_steps=5, total_timeout=60.0, llm_timeout=10.0)
    assert cfg.max_steps == 5

    # Total timeout cannot be smaller than llm_timeout
    with pytest.raises(AgentConfigurationError):
        AgentConfig(total_timeout=5.0, llm_timeout=10.0)
