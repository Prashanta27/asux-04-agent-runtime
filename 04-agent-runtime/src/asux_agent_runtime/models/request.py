from typing import Any

from asux_tool_runtime.models.context import PermissionContext
from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """
    Validated request to execute an Agent run toward a user goal.
    Caller identity and permissions are strictly passed via trusted PermissionContext.
    """

    goal: str = Field(..., min_length=1, description="The user's goal or request.")
    conversation_id: str | None = Field(
        default=None, description="Optional conversation or session identifier."
    )
    context: PermissionContext = Field(
        ...,
        description="Trusted caller and tenant context. Cannot be overwritten by user input.",
    )
    max_steps: int | None = Field(
        default=None,
        gt=0,
        description="Optional per-request step limit. Cannot exceed global safety limits.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Safe custom metadata for tracking or observability."
    )
