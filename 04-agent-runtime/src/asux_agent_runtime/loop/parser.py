import json
import re
from typing import Any

from pydantic import ValidationError

from asux_agent_runtime.exceptions.errors import AgentDecisionParsingError
from asux_agent_runtime.models.decision import AgentDecision


class DecisionParser:
    """
    Parses and strictly validates model responses into structured AgentDecision instances.
    Operates strictly on observable actions and decisions.
    """

    @classmethod
    def parse(cls, content: str | None) -> AgentDecision:
        """
        Parses raw text or JSON response into a validated AgentDecision.

        Raises:
            AgentDecisionParsingError: If content is missing, malformed JSON,
                or fails schema validation.
        """
        if not content or not content.strip():
            raise AgentDecisionParsingError("Model returned empty or null content.")

        cleaned = content.strip()

        # Handle Markdown code block formatting if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Try parsing directly
        data: Any = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as decode_err:
            # Fallback: Extract the first outermost JSON object in the string
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError as err:
                    raise AgentDecisionParsingError(
                        f"Malformed JSON in model response: {err.msg}"
                    ) from err
            else:
                raise AgentDecisionParsingError(
                    "Model response does not contain a valid JSON object."
                ) from decode_err

        if not isinstance(data, dict):
            raise AgentDecisionParsingError(
                f"Expected JSON object in model response, got {type(data).__name__}."
            )

        try:
            return AgentDecision.model_validate(data)
        except ValidationError as e:
            raise AgentDecisionParsingError(f"Decision schema validation failed: {e}") from e
