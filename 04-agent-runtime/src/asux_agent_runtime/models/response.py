from typing import Any

from pydantic import BaseModel, Field

from asux_agent_runtime.models.status import AgentStatus


class ToolInvocationRecord(BaseModel):
    """
    Record of a tool invocation executed during an agent run.
    """

    tool_name: str = Field(..., description="Name of the invoked tool.")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool."
    )
    success: bool = Field(..., description="Whether the tool execution succeeded.")
    duration_ms: float = Field(default=0.0, description="Duration in milliseconds.")
    error: str | None = Field(default=None, description="Safe error message if execution failed.")


class AgentResponse(BaseModel):
    """
    Structured outcome returned to the caller upon completion, clarification, or failure.
    Internal stack traces and secrets are strictly excluded.
    """

    run_id: str = Field(..., description="Unique run identifier.")
    conversation_id: str | None = Field(
        default=None, description="Session or conversation identifier."
    )
    final_response: str | None = Field(
        default=None, description="User-facing final response or clarification question."
    )
    status: AgentStatus = Field(..., description="Final status of the agent run.")
    steps_executed: int = Field(default=0, description="Total number of steps executed.")
    tool_calls_executed: int = Field(default=0, description="Total number of tool calls executed.")
    tool_invocations: list[ToolInvocationRecord] = Field(
        default_factory=list, description="List of safe tool invocation summaries."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Safe metadata regarding execution metrics."
    )
    termination_reason: str | None = Field(
        default=None, description="Safe reason for termination or failure if applicable."
    )
