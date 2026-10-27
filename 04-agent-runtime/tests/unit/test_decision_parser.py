import pytest

from asux_agent_runtime.exceptions.errors import AgentDecisionParsingError
from asux_agent_runtime.loop.parser import DecisionParser
from asux_agent_runtime.models.decision import AgentAction


def test_parse_valid_final() -> None:
    raw = '{"action": "final", "message": "Here is the final answer."}'
    decision = DecisionParser.parse(raw)
    assert decision.action == AgentAction.FINAL
    assert decision.message == "Here is the final answer."


def test_parse_valid_clarification() -> None:
    raw = '{"action": "clarification", "message": "Please specify your order ID."}'
    decision = DecisionParser.parse(raw)
    assert decision.action == AgentAction.CLARIFICATION
    assert decision.message == "Please specify your order ID."


def test_parse_valid_tool_call() -> None:
    raw = (
        '{"action": "tool_call", "tool_name": "product_search", "arguments": {"query": "monitor"}}'
    )
    decision = DecisionParser.parse(raw)
    assert decision.action == AgentAction.TOOL_CALL
    assert decision.tool_name == "product_search"
    assert decision.arguments == {"query": "monitor"}


def test_parse_markdown_fenced_json() -> None:
    raw = """```json
    {
      "action": "final",
      "message": "Fenced JSON response."
    }
    ```"""
    decision = DecisionParser.parse(raw)
    assert decision.action == AgentAction.FINAL
    assert decision.message == "Fenced JSON response."


def test_parse_embedded_json() -> None:
    raw = 'Here is my decision: {"action": "final", "message": "Embedded JSON"}'
    decision = DecisionParser.parse(raw)
    assert decision.action == AgentAction.FINAL
    assert decision.message == "Embedded JSON"


def test_parse_empty_content() -> None:
    with pytest.raises(AgentDecisionParsingError, match="empty or null"):
        DecisionParser.parse("")

    with pytest.raises(AgentDecisionParsingError, match="empty or null"):
        DecisionParser.parse(None)


def test_parse_malformed_json() -> None:
    with pytest.raises(AgentDecisionParsingError, match=r"Malformed JSON|does not contain"):
        DecisionParser.parse("{action: invalid_json, message")


def test_parse_invalid_schema() -> None:
    # Missing required message for final
    with pytest.raises(AgentDecisionParsingError, match="validation failed"):
        DecisionParser.parse('{"action": "final"}')

    # Tool call missing arguments
    with pytest.raises(AgentDecisionParsingError, match="validation failed"):
        DecisionParser.parse('{"action": "tool_call", "tool_name": "order_lookup"}')
