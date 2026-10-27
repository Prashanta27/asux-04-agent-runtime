import json
import time
import uuid
from collections.abc import AsyncGenerator

from asux_llm_gateway.clients.llm_client import LLMClient
from asux_llm_gateway.exceptions.errors import GatewayError, GatewayTimeoutError
from asux_llm_gateway.models.messages import Message, MessageRole
from asux_llm_gateway.models.requests import GatewayRequest
from asux_prompt_engine.models.rendering import RenderRequest
from asux_prompt_engine.services.prompt_engine import PromptEngine
from asux_tool_runtime.executor.tool_executor import ToolExecutor

from asux_agent_runtime.config import AgentConfig
from asux_agent_runtime.exceptions.errors import (
    AgentDecisionParsingError,
    AgentValidationError,
)
from asux_agent_runtime.loop.parser import DecisionParser
from asux_agent_runtime.models.decision import AgentAction
from asux_agent_runtime.models.events import AgentEvent, AgentEventType
from asux_agent_runtime.models.request import AgentRequest
from asux_agent_runtime.models.response import AgentResponse, ToolInvocationRecord
from asux_agent_runtime.models.state import AgentState, AgentStep
from asux_agent_runtime.models.status import AgentStatus
from asux_agent_runtime.observability.logger import get_logger

logger = get_logger(__name__)


class AgentLoop:
    """
    Controlled execution loop for a single AI Assistant.
    Coordinates between Prompt Engine, LLM Gateway, and Tool Runtime.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_engine: PromptEngine,
        tool_executor: ToolExecutor,
        config: AgentConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_engine = prompt_engine
        self.tool_executor = tool_executor
        self.config = config or AgentConfig()

    def _format_available_tools(self) -> str:
        """
        Formats definitions of all registered and enabled tools for inclusion in prompt.
        """
        tools = self.tool_executor.registry.list_definitions(include_disabled=False)
        if not tools:
            return "No tools are currently registered or enabled."

        tool_descriptions: list[str] = []
        for tool in tools:
            schema_json = json.dumps(tool.input_schema.model_json_schema())
            tool_descriptions.append(
                f"- Tool: {tool.name}\n"
                f"  Description: {tool.description}\n"
                f"  Required Permissions: {tool.required_permissions}\n"
                f"  Input Schema: {schema_json}"
            )
        return "\n\n".join(tool_descriptions)

    def _truncate_output(self, content: str) -> str:
        if len(content) > self.config.max_tool_result_chars:
            return (
                content[: self.config.max_tool_result_chars] + "... [TRUNCATED DUE TO SIZE LIMIT]"
            )
        return content

    async def run(self, request: AgentRequest) -> AgentResponse:
        """
        Executes the agent loop to completion, returning the structured AgentResponse.
        """
        events: list[AgentEvent] = []
        async for event in self.run_stream(request):
            events.append(event)

        # Retrieve the final response payload from the terminal event
        terminal_event = events[-1] if events else None
        if terminal_event and "response" in terminal_event.data:
            return AgentResponse.model_validate(terminal_event.data["response"])

        raise RuntimeError("Agent loop ended unexpectedly without emitting a final response.")

    async def run_stream(self, request: AgentRequest) -> AsyncGenerator[AgentEvent, None]:
        """
        Executes the agent loop, yielding observable progress events.
        """
        # 1. Validate request
        if not request.goal or not request.goal.strip():
            raise AgentValidationError("AgentRequest goal cannot be empty or blank.")

        # Determine effective limits (request cannot override global max)
        effective_max_steps = self.config.max_steps
        if request.max_steps and request.max_steps > 0:
            effective_max_steps = min(request.max_steps, self.config.max_steps)

        effective_max_tool_calls = self.config.max_tool_calls
        run_id = f"run_{uuid.uuid4().hex[:12]}"

        # 2. Initialize State
        state = AgentState(
            run_id=run_id,
            conversation_id=request.conversation_id,
            goal=request.goal.strip(),
            status=AgentStatus.RUNNING,
            start_time=time.monotonic(),
            metadata=request.metadata.copy(),
        )

        yield AgentEvent(
            event_type=AgentEventType.RUN_STARTED,
            run_id=run_id,
            step=0,
            data={"goal": state.goal, "conversation_id": state.conversation_id},
        )

        # 3. Discover Tools and Render Prompt
        tools_schema_str = self._format_available_tools()
        render_req = RenderRequest(
            name=self.config.prompt_name,
            version=self.config.prompt_version,
            variables={
                "tools_schema": tools_schema_str,
                "tenant_id": request.context.tenant_id,
            },
        )

        render_res = self.prompt_engine.render(render_req)
        state.messages.extend(render_res.messages)
        state.messages.append(Message.user(content=state.goal))

        final_response_text: str | None = None

        # 4. Controlled Agent Execution Loop
        while state.step_count < effective_max_steps:
            step_start = time.monotonic()
            step_num = state.step_count + 1

            # Check overall run deadline
            if (time.monotonic() - state.start_time) > self.config.total_timeout:
                state.status = AgentStatus.FAILED
                state.termination_reason = (
                    f"Execution exceeded total run deadline of {self.config.total_timeout}s."
                )
                break

            # Enforce maximum retained messages safeguard (preserving initial system prompt & goal)
            if len(state.messages) > self.config.max_retained_messages:
                system_msg = state.messages[0]
                user_goal = state.messages[1]
                recent_msgs = state.messages[-(self.config.max_retained_messages - 2) :]
                state.messages = [system_msg, user_goal, *recent_msgs]

            yield AgentEvent(
                event_type=AgentEventType.MODEL_CALL_STARTED,
                run_id=run_id,
                step=step_num,
                data={"model": self.config.model},
            )

            # Call LLM Gateway
            gateway_req = GatewayRequest(
                model=self.config.model,
                messages=state.messages,
                timeout=self.config.llm_timeout,
            )

            try:
                gateway_res = await self.llm_client.complete(gateway_req)
                model_output = gateway_res.content
            except (TimeoutError, GatewayTimeoutError):
                state.consecutive_failure_count += 1
                logger.warning(
                    f"LLM call timed out for {run_id} ({state.consecutive_failure_count})."
                )
                if state.consecutive_failure_count >= self.config.max_consecutive_failures:
                    state.status = AgentStatus.FAILED
                    state.termination_reason = (
                        "Aborted: LLM call exceeded maximum consecutive timeout limits."
                    )
                    break
                state.messages.append(
                    Message.user(
                        content="Previous model call timed out. Please provide a concise decision."
                    )
                )
                continue
            except GatewayError as e:
                state.consecutive_failure_count += 1
                logger.warning(f"LLM gateway error: {e}")
                if state.consecutive_failure_count >= self.config.max_consecutive_failures:
                    state.status = AgentStatus.FAILED
                    state.termination_reason = f"Aborted due to repeated LLM errors: {e}"
                    break
                state.messages.append(
                    Message.user(
                        content="Provider error occurred. Please try again with a valid decision."
                    )
                )
                continue

            # Parse and Validate Decision
            try:
                decision = DecisionParser.parse(model_output)
            except AgentDecisionParsingError as parse_err:
                state.consecutive_failure_count += 1
                logger.warning(
                    f"Model decision parse failed in run {run_id}: {parse_err} "
                    f"({state.consecutive_failure_count}/{self.config.max_consecutive_failures})"
                )
                if state.consecutive_failure_count >= self.config.max_consecutive_failures:
                    state.status = AgentStatus.FAILED
                    state.termination_reason = (
                        f"Aborted: Exceeded max consecutive decision parsing failures "
                        f"({self.config.max_consecutive_failures}). Last error: {parse_err}"
                    )
                    break

                # Bounded recovery feedback
                state.messages.append(Message.assistant(content=model_output))
                state.messages.append(
                    Message.user(
                        content=(
                            f"Decision could not be parsed: {parse_err}. "
                            "You must strictly output a valid JSON decision."
                        )
                    )
                )
                continue

            yield AgentEvent(
                event_type=AgentEventType.MODEL_DECISION_RECEIVED,
                run_id=run_id,
                step=step_num,
                data={"action": decision.action.value},
            )

            # Process Decision
            if decision.action == AgentAction.FINAL:
                state.status = AgentStatus.COMPLETED
                final_response_text = decision.message
                step_duration = (time.monotonic() - step_start) * 1000
                state.steps.append(
                    AgentStep(step_number=step_num, decision=decision, duration_ms=step_duration)
                )
                state.step_count += 1

                yield AgentEvent(
                    event_type=AgentEventType.FINAL_RESPONSE_AVAILABLE,
                    run_id=run_id,
                    step=step_num,
                    data={"response": final_response_text},
                )
                break

            elif decision.action == AgentAction.CLARIFICATION:
                state.status = AgentStatus.NEEDS_INPUT
                final_response_text = decision.message
                step_duration = (time.monotonic() - step_start) * 1000
                state.steps.append(
                    AgentStep(step_number=step_num, decision=decision, duration_ms=step_duration)
                )
                state.step_count += 1

                yield AgentEvent(
                    event_type=AgentEventType.CLARIFICATION_REQUIRED,
                    run_id=run_id,
                    step=step_num,
                    data={"question": final_response_text},
                )
                break

            elif decision.action == AgentAction.TOOL_CALL:
                # Check tool call budget
                if state.tool_call_count >= effective_max_tool_calls:
                    state.status = AgentStatus.MAX_STEPS_REACHED
                    state.termination_reason = (
                        f"Exceeded maximum allowed tool calls limit of {effective_max_tool_calls}."
                    )
                    break

                tool_name = decision.tool_name or ""
                arguments = decision.arguments or {}

                # Duplicate call safeguard
                if self.config.duplicate_call_safeguard and state.tool_invocations:
                    last_inv = state.tool_invocations[-1]
                    if (
                        last_inv.tool_name == tool_name
                        and last_inv.arguments == arguments
                        and not last_inv.success
                    ):
                        state.consecutive_failure_count += 1
                        logger.warning(f"Duplicate failed call for '{tool_name}' in run {run_id}.")
                        if state.consecutive_failure_count >= self.config.max_consecutive_failures:
                            state.status = AgentStatus.FAILED
                            state.termination_reason = (
                                f"Aborted: Duplicate consecutive failed call to tool '{tool_name}'."
                            )
                            break
                    else:
                        state.consecutive_failure_count = 0

                yield AgentEvent(
                    event_type=AgentEventType.TOOL_CALL_REQUESTED,
                    run_id=run_id,
                    step=step_num,
                    data={"tool_name": tool_name},
                )

                inv_id = f"inv_{uuid.uuid4().hex[:8]}"

                # Append assistant tool call message to state
                assistant_tool_msg = json.dumps(
                    {"action": "tool_call", "tool_name": tool_name, "arguments": arguments}
                )
                state.messages.append(Message.assistant(content=assistant_tool_msg))

                # Execute tool strictly via Tool Runtime (Project 3)
                tool_res = await self.tool_executor.execute(
                    tool_name=tool_name,
                    arguments=arguments,
                    context=request.context,
                    invocation_id=inv_id,
                )

                # Record invocation
                inv_record = ToolInvocationRecord(
                    tool_name=tool_name,
                    arguments=arguments,
                    success=tool_res.success,
                    duration_ms=tool_res.duration_ms,
                    error=tool_res.error,
                )
                state.tool_invocations.append(inv_record)
                state.tool_call_count += 1

                # Emit execution result event
                if tool_res.success:
                    state.consecutive_failure_count = 0
                    yield AgentEvent(
                        event_type=AgentEventType.TOOL_EXECUTION_COMPLETED,
                        run_id=run_id,
                        step=step_num,
                        data={"tool_name": tool_name, "duration_ms": tool_res.duration_ms},
                    )
                    # Format output data
                    if tool_res.data is None:
                        data_str = "null"
                    elif hasattr(tool_res.data, "model_dump_json"):
                        data_str = str(tool_res.data.model_dump_json())
                    elif isinstance(tool_res.data, (dict, list)):
                        data_str = json.dumps(tool_res.data)
                    else:
                        data_str = str(tool_res.data)

                    formatted_data = self._truncate_output(data_str)
                    state.messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=formatted_data,
                            name=tool_name,
                            tool_call_id=inv_id,
                        )
                    )
                else:
                    state.consecutive_failure_count += 1
                    yield AgentEvent(
                        event_type=AgentEventType.TOOL_EXECUTION_FAILED,
                        run_id=run_id,
                        step=step_num,
                        data={"tool_name": tool_name, "error": tool_res.error},
                    )
                    err_msg = (
                        f"Tool '{tool_name}' execution failed: {tool_res.error}. "
                        "If permission was denied, do NOT retry this tool call."
                    )
                    state.messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=err_msg,
                            name=tool_name,
                            tool_call_id=inv_id,
                        )
                    )

                step_duration = (time.monotonic() - step_start) * 1000
                state.steps.append(
                    AgentStep(
                        step_number=step_num,
                        decision=decision,
                        tool_result=tool_res,
                        duration_ms=step_duration,
                    )
                )
                state.step_count += 1

        # Check if terminated by step exhaustion
        if state.status == AgentStatus.RUNNING:
            state.status = AgentStatus.MAX_STEPS_REACHED
            state.termination_reason = (
                f"Agent reached maximum reasoning step limit ({effective_max_steps}) "
                "without completion."
            )

        state.elapsed_duration_ms = (time.monotonic() - state.start_time) * 1000

        # Build response
        final_response = AgentResponse(
            run_id=state.run_id,
            conversation_id=state.conversation_id,
            final_response=final_response_text,
            status=state.status,
            steps_executed=state.step_count,
            tool_calls_executed=state.tool_call_count,
            tool_invocations=state.tool_invocations,
            metadata=state.metadata,
            termination_reason=state.termination_reason,
        )

        yield AgentEvent(
            event_type=AgentEventType.RUN_TERMINATED,
            run_id=run_id,
            step=state.step_count,
            data={"status": state.status.value, "response": final_response.model_dump()},
        )
