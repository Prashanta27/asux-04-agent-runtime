# Agent Loop Specification

The `AgentLoop` executes a bounded, sequential cycle until a terminal status is achieved.

## Decision Actions
The model must output a valid JSON decision adhering to one of the following schemas:

### 1. `tool_call`
```json
{
  "action": "tool_call",
  "tool_name": "product_search",
  "arguments": {
    "query": "laptop"
  }
}
```

### 2. `clarification`
```json
{
  "action": "clarification",
  "message": "Could you please provide your order ID so I can look up the shipment status?"
}
```

### 3. `final`
```json
{
  "action": "final",
  "message": "We have 1 product matching 'laptop' priced at $99.99."
}
```

## Step Flow
1. **Validate Request:** Verify `goal` is non-empty and context is populated.
2. **Context Discovery:** Discover registered tools and format JSON schemas.
3. **Render Prompt:** Fetch template from `PromptEngine` with dynamic tool definitions.
4. **Call LLM:** Dispatch request to `LLMClient`.
5. **Parse Decision:** `DecisionParser` validates JSON schema. If invalid, allows bounded recovery (up to `max_consecutive_failures`).
6. **Execute Action:**
   - If `final`: status `COMPLETED`, return final text.
   - If `clarification`: status `NEEDS_INPUT`, return clarification question.
   - If `tool_call`: dispatch to `ToolExecutor.execute()`. Record invocation in history.
7. **Safeguards Evaluation:** Check timeouts, duplicate calls, and step limits.
