from enum import StrEnum


class AgentStatus(StrEnum):
    """
    Explicit lifecycle and termination statuses for an Agent run.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    NEEDS_INPUT = "needs_input"
    MAX_STEPS_REACHED = "max_steps_reached"
    FAILED = "failed"
    CANCELLED = "cancelled"
