# Error Handling and Recovery Strategy

The Agent Runtime distinguishes recoverable from non-recoverable failures:

## Recoverable Failures
1. **Decision JSON Parsing Error:** If the model outputs invalid JSON or missing fields, the runtime records the error and prompts the model to correct its format within `max_consecutive_failures`.
2. **Tool Input Validation Error:** If the model sends wrong arguments (e.g. missing required field), the Tool Runtime rejects the call and returns a structured error. The model sees this error and can adjust arguments in the next step.
3. **Transient LLM Timeout:** If a single call times out, the loop provides bounded recovery before aborting.

## Non-Recoverable Failures
1. **Authorization Denial (`ToolAuthorizationError`):** When a caller lacks permissions, the Tool Runtime immediately blocks execution. The agent is explicitly told permission was denied and instructed not to retry.
2. **Run Deadline Exceeded:** If `total_timeout` is breached, the loop terminates immediately with `status="failed"`.
3. **Budget Exhaustion:** If `max_steps` or `max_tool_calls` is reached, execution halts with `status="max_steps_reached"`.
