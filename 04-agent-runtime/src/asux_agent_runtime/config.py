from pydantic import BaseModel, Field, model_validator

from asux_agent_runtime.exceptions.errors import AgentConfigurationError


class AgentConfig(BaseModel):
    """
    Centralized configuration and safety boundaries for the Agent Runtime.
    """

    # Model and Prompt
    model: str = Field(default="gpt-4o-mini", description="Target model name in LLM Gateway.")
    prompt_name: str = Field(
        default="commerce_assistant", description="Prompt template name in Prompt Engine."
    )
    prompt_version: str = Field(
        default="1.0.0", description="Prompt template version in Prompt Engine."
    )

    # Execution and Budget Limits
    max_steps: int = Field(default=10, gt=0, le=50, description="Global maximum reasoning steps.")
    max_tool_calls: int = Field(
        default=10, gt=0, le=50, description="Global maximum tool invocations per run."
    )
    max_consecutive_failures: int = Field(
        default=3,
        gt=0,
        le=10,
        description="Maximum consecutive recoverable failures before aborting.",
    )

    # Timeouts (in seconds)
    llm_timeout: float = Field(
        default=30.0, gt=0.0, le=300.0, description="Per-LLM-call timeout in seconds."
    )
    tool_timeout: float = Field(
        default=30.0, gt=0.0, le=300.0, description="Per-tool-call timeout in seconds."
    )
    total_timeout: float = Field(
        default=120.0, gt=0.0, le=600.0, description="Total run deadline timeout in seconds."
    )

    # Context and Content Safeguards
    max_retained_messages: int = Field(
        default=40, gt=2, description="Maximum conversation messages retained in context."
    )
    max_tool_result_chars: int = Field(
        default=4000, gt=100, description="Maximum characters allowed in serialized tool output."
    )
    duplicate_call_safeguard: bool = Field(
        default=True,
        description="Prevent consecutive identical tool calls to mitigate infinite loops.",
    )

    @model_validator(mode="after")
    def validate_timeouts(self) -> "AgentConfig":
        if self.total_timeout < self.llm_timeout:
            raise AgentConfigurationError("total_timeout cannot be smaller than llm_timeout.")
        return self
