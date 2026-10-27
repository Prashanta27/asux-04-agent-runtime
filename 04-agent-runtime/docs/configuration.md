# Configuration Reference: AgentConfig

`AgentConfig` defines global safety constraints and model defaults.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `str` | `"gpt-4o-mini"` | Target model routed by LLM Gateway. |
| `prompt_name` | `str` | `"commerce_assistant"` | Template name in Prompt Engine. |
| `prompt_version` | `str` | `"1.0.0"` | Template version in Prompt Engine. |
| `max_steps` | `int` | `10` | Maximum agent reasoning steps allowed. |
| `max_tool_calls` | `int` | `10` | Maximum tool calls allowed per run. |
| `max_consecutive_failures` | `int` | `3` | Maximum consecutive failures before aborting. |
| `llm_timeout` | `float` | `30.0` | Timeout per LLM call in seconds. |
| `tool_timeout` | `float` | `30.0` | Timeout per tool invocation in seconds. |
| `total_timeout` | `float` | `120.0` | Total run deadline in seconds. |
| `max_retained_messages` | `int` | `40` | Context window truncation threshold. |
| `max_tool_result_chars` | `int` | `4000` | Tool output truncation limit in chars. |
| `duplicate_call_safeguard` | `bool` | `True` | Prevents consecutive identical failed calls. |
