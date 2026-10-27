import pytest
from asux_tool_runtime.executor.tool_executor import ToolExecutor
from asux_tool_runtime.models.context import PermissionContext


@pytest.mark.asyncio
async def test_tool_runtime_discovery_and_execution(
    tool_executor: ToolExecutor,
    full_permission_context: PermissionContext,
) -> None:
    tools = tool_executor.registry.list_definitions()
    tool_names = [t.name for t in tools]
    assert "product_search" in tool_names
    assert "customer_lookup" in tool_names
    assert "create_task" in tool_names

    # Execution with full permissions
    res = await tool_executor.execute(
        tool_name="product_search",
        arguments={"query": "keyboard"},
        context=full_permission_context,
    )
    assert res.success is True
    assert res.error is None
    assert res.data is not None


@pytest.mark.asyncio
async def test_tool_runtime_permission_boundary(
    tool_executor: ToolExecutor,
    read_only_permission_context: PermissionContext,
) -> None:
    # read_only has only "product:read", but "create_task" requires "task:write"
    res = await tool_executor.execute(
        tool_name="create_task",
        arguments={"title": "Test Task"},
        context=read_only_permission_context,
    )
    assert res.success is False
    assert "lacks required permission" in (res.error or "")


@pytest.mark.asyncio
async def test_tool_runtime_invalid_arguments(
    tool_executor: ToolExecutor,
    full_permission_context: PermissionContext,
) -> None:
    # "customer_lookup" requires "customer_id", missing here
    res = await tool_executor.execute(
        tool_name="customer_lookup",
        arguments={},
        context=full_permission_context,
    )
    assert res.success is False
    assert "Invalid arguments" in (res.error or "")
