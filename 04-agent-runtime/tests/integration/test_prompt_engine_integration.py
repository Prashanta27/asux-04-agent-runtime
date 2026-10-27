import pytest
from asux_prompt_engine.exceptions.errors import PromptNotFoundError
from asux_prompt_engine.models.rendering import RenderRequest
from asux_prompt_engine.services.prompt_engine import PromptEngine


def test_prompt_engine_renders_commerce_assistant(prompt_engine: PromptEngine) -> None:
    req = RenderRequest(
        name="commerce_assistant",
        version="1.0.0",
        variables={
            "tools_schema": "- Tool: product_search\n  Description: Searches products",
            "tenant_id": "test_tenant",
        },
    )

    result = prompt_engine.render(req)
    assert result.prompt_name == "commerce_assistant"
    assert len(result.messages) == 1
    content = result.messages[0].content or ""
    assert "AsuX AI Commerce Assistant" in content
    assert "product_search" in content


def test_prompt_engine_nonexistent_prompt(prompt_engine: PromptEngine) -> None:
    req = RenderRequest(
        name="non_existent_prompt",
        version="1.0.0",
        variables={"tools_schema": "none"},
    )
    with pytest.raises(PromptNotFoundError):
        prompt_engine.render(req)
