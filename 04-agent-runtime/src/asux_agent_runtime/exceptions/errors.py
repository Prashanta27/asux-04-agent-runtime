class AgentRuntimeError(Exception):
    """Base exception for all errors raised by the Agent Runtime."""


class AgentValidationError(AgentRuntimeError):
    """Raised when an agent request or runtime input fails schema or logical validation."""


class AgentConfigurationError(AgentRuntimeError):
    """Raised when the agent runtime or loop is configured invalidly."""


class AgentDecisionParsingError(AgentRuntimeError):
    """Raised when a model output cannot be parsed into a valid structured AgentDecision."""


class AgentExecutionLimitError(AgentRuntimeError):
    """Raised when an execution limit (e.g. max steps, tool calls, or failures) is exceeded."""


class AgentTimeoutError(AgentRuntimeError):
    """Raised when the agent run or a component exceeds its configured execution deadline."""


class AgentStateError(AgentRuntimeError):
    """Raised when an invalid state transition or state corruption occurs during execution."""
