from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from asux_llm_gateway.clients.llm_client import LLMClient
from asux_prompt_engine.registry.file_registry import FilePromptRegistry
from asux_prompt_engine.services.prompt_engine import PromptEngine
from asux_tool_runtime.executor.tool_executor import ToolExecutor
from asux_tool_runtime.models.context import PermissionContext
from asux_tool_runtime.registry.tool_registry import InMemoryToolRegistry
from asux_tool_runtime.tools.catalog import register_builtin_tools

from asux_agent_runtime.config import AgentConfig


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    client = AsyncMock(spec=LLMClient)
    return client


@pytest.fixture
def tool_registry() -> InMemoryToolRegistry:
    registry = InMemoryToolRegistry()
    register_builtin_tools(registry)
    return registry


@pytest.fixture
def tool_executor(tool_registry: InMemoryToolRegistry) -> ToolExecutor:
    return ToolExecutor(registry=tool_registry, default_timeout=5.0)


@pytest.fixture
def prompt_engine() -> PromptEngine:
    # Use prompts directory in 04-agent-runtime/src/asux_agent_runtime/prompts
    prompts_dir = Path(__file__).parent.parent / "src" / "asux_agent_runtime" / "prompts"
    registry = FilePromptRegistry(storage_dir=prompts_dir)
    return PromptEngine(registry=registry)


@pytest.fixture
def default_config() -> AgentConfig:
    return AgentConfig(
        max_steps=5,
        max_tool_calls=5,
        max_consecutive_failures=2,
        llm_timeout=5.0,
        tool_timeout=5.0,
        total_timeout=15.0,
    )


@pytest.fixture
def full_permission_context() -> PermissionContext:
    return PermissionContext(
        principal_id="user_admin",
        principal_type="user",
        tenant_id="tenant_001",
        permissions=[
            "customer:read",
            "product:read",
            "order:read",
            "conversation:read",
            "task:write",
            "message:write",
        ],
    )


@pytest.fixture
def read_only_permission_context() -> PermissionContext:
    return PermissionContext(
        principal_id="user_guest",
        principal_type="user",
        tenant_id="tenant_001",
        permissions=["product:read"],
    )
