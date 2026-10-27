from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class AgentAction(StrEnum):
    """
    Standardized, provider-independent actions an agent decision can represent.
    """

    FINAL = "final"
    TOOL_CALL = "tool_call"
    CLARIFICATION = "clarification"


class AgentDecision(BaseModel):
    """
    Validated decision returned by the model during an agent step.
    Operates strictly on observable actions and decisions without hidden chain-of-thought.
    """

    action: AgentAction = Field(
        ..., description="The action to take: 'final', 'tool_call', or 'clarification'."
    )
    tool_name: str | None = Field(
        default=None, description="The name of the tool to execute if action is 'tool_call'."
    )
    arguments: dict[str, Any] | None = Field(
        default=None, description="The JSON arguments for the tool if action is 'tool_call'."
    )
    message: str | None = Field(
        default=None,
        description="The user-facing message for 'final' or question for 'clarification'.",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "AgentDecision":
        if self.action == AgentAction.TOOL_CALL:
            if not self.tool_name or not self.tool_name.strip():
                raise ValueError("Decision action 'tool_call' requires a valid 'tool_name'.")
            if self.arguments is None or not isinstance(self.arguments, dict):
                raise ValueError("Decision action 'tool_call' requires dictionary 'arguments'.")
        elif self.action in (AgentAction.FINAL, AgentAction.CLARIFICATION):
            if not self.message or not self.message.strip():
                raise ValueError(
                    f"Decision action '{self.action.value}' requires a non-empty 'message'."
                )
        return self
