import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentEventType(StrEnum):
    """
    Event types emitted during streaming execution of an agent run.
    """

    RUN_STARTED = "run_started"
    MODEL_CALL_STARTED = "model_call_started"
    MODEL_DECISION_RECEIVED = "model_decision_received"
    TOOL_CALL_REQUESTED = "tool_call_requested"
    TOOL_EXECUTION_COMPLETED = "tool_execution_completed"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    CLARIFICATION_REQUIRED = "clarification_required"
    FINAL_RESPONSE_AVAILABLE = "final_response_available"
    RUN_TERMINATED = "run_terminated"


class AgentEvent(BaseModel):
    """
    Observable event emitted by the agent loop during execution.
    """

    event_type: AgentEventType = Field(..., description="The type of event.")
    run_id: str = Field(..., description="Identifier of the executing run.")
    step: int = Field(default=0, description="The current step index.")
    data: dict[str, Any] = Field(default_factory=dict, description="Safe event payload.")
    timestamp: float = Field(
        default_factory=time.time, description="Unix timestamp when event occurred."
    )
