from typing import Any

from asux_llm_gateway.models.messages import Message
from asux_tool_runtime.models.result import ToolResult
from pydantic import BaseModel, Field

from asux_agent_runtime.models.decision import AgentDecision
from asux_agent_runtime.models.response import ToolInvocationRecord
from asux_agent_runtime.models.status import AgentStatus


class AgentStep(BaseModel):
    """
    Representation of an individual reasoning and action step in the agent loop.
    """

    step_number: int = Field(..., description="1-indexed step number.")
    decision: AgentDecision = Field(..., description="The decision made by the model.")
    tool_result: ToolResult | None = Field(
        default=None, description="Result of tool execution if action was tool_call."
    )
    duration_ms: float = Field(default=0.0, description="Duration of this step in milliseconds.")


class AgentState(BaseModel):
    """
    In-memory state tracking for an isolated agent execution run.
    """

    run_id: str = Field(..., description="Unique run identifier.")
    conversation_id: str | None = Field(
        default=None, description="Optional conversation or session ID."
    )
    goal: str = Field(..., description="The original user goal.")
    status: AgentStatus = Field(
        default=AgentStatus.PENDING, description="Current lifecycle status."
    )

    messages: list[Message] = Field(
        default_factory=list, description="Ordered conversation history passed to the LLM Gateway."
    )
    steps: list[AgentStep] = Field(
        default_factory=list, description="Historical list of all completed steps."
    )

    step_count: int = Field(default=0, description="Total steps completed so far.")
    tool_call_count: int = Field(default=0, description="Total tool calls executed so far.")
    consecutive_failure_count: int = Field(
        default=0, description="Count of consecutive recoverable failures."
    )
    tool_invocations: list[ToolInvocationRecord] = Field(
        default_factory=list, description="Historical tool invocation records."
    )

    start_time: float = Field(default=0.0, description="Monotonic start timestamp.")
    elapsed_duration_ms: float = Field(
        default=0.0, description="Total elapsed runtime in milliseconds."
    )
    termination_reason: str | None = Field(
        default=None, description="Explanation when terminating prematurely or with an error."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Custom safe execution metadata."
    )
