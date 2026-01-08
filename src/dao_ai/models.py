import uuid
from os import PathLike
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Generator,
    Literal,
    Optional,
    Sequence,
    Union,
)

from databricks_langchain import ChatDatabricks

if TYPE_CHECKING:
    pass

# Import official LangChain HITL TypedDict definitions
# Reference: https://docs.langchain.com/oss/python/langchain/human-in-the-loop
from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    Decision,
    EditDecision,
    HITLRequest,
    RejectDecision,
    ReviewConfig,
)
from langchain_community.adapters.openai import convert_openai_messages
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Interrupt, StateSnapshot
from loguru import logger
from mlflow import MlflowClient
from mlflow.pyfunc import ChatAgent, ChatModel, ResponsesAgent
from mlflow.types.agent import ChatContext
from mlflow.types.llm import (
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
    ChatParams,
)
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.types.responses_helpers import (
    Message,
    ResponseInputTextParam,
)
from pydantic import BaseModel, Field, create_model

from dao_ai.messages import (
    has_langchain_messages,
    has_mlflow_messages,
    has_mlflow_responses_messages,
    last_human_message,
)
from dao_ai.state import Context


def get_latest_model_version(model_name: str) -> int:
    """
    Retrieve the latest version number of a registered MLflow model.

    Queries the MLflow Model Registry to find the highest version number
    for a given model name, which is useful for ensuring we're using
    the most recent model version.

    Args:
        model_name: The name of the registered model in MLflow

    Returns:
        The latest version number as an integer
    """
    mlflow_client: MlflowClient = MlflowClient()
    latest_version: int = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int: int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


def is_interrupted(snapshot: StateSnapshot) -> bool:
    """
    Check if the graph state is currently interrupted (paused for human-in-the-loop).

    Based on LangChain documentation:
    - StateSnapshot has an `interrupts` attribute which is a tuple
    - When interrupted, the tuple contains Interrupt objects
    - When not interrupted, it's an empty tuple ()

    Args:
        snapshot: The StateSnapshot to check

    Returns:
        True if the graph is interrupted (has pending HITL actions), False otherwise

    Example:
        >>> snapshot = await graph.aget_state(config)
        >>> if is_interrupted(snapshot):
        ...     print("Graph is waiting for human input")
    """
    # Check if snapshot has any interrupts
    # According to LangChain docs, interrupts is a tuple that's empty () when no interrupts
    return bool(snapshot.interrupts)


async def get_state_snapshot_async(
    graph: CompiledStateGraph, thread_id: str
) -> Optional[StateSnapshot]:
    """
    Retrieve the state snapshot from the graph for a given thread_id asynchronously.

    This utility function accesses the graph's checkpointer to retrieve the current
    state snapshot, which contains the full state values and metadata.

    Args:
        graph: The compiled LangGraph state machine
        thread_id: The thread/conversation ID to retrieve state for

    Returns:
        StateSnapshot if found, None otherwise
    """
    logger.trace("Retrieving state snapshot", thread_id=thread_id)
    try:
        # Check if graph has a checkpointer
        if graph.checkpointer is None:
            logger.trace("No checkpointer available in graph")
            return None

        # Get the current state from the checkpointer (use async version)
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        state_snapshot: Optional[StateSnapshot] = await graph.aget_state(config)

        if state_snapshot is None:
            logger.trace("No state found for thread", thread_id=thread_id)
            return None

        return state_snapshot

    except Exception as e:
        logger.warning(
            "Error retrieving state snapshot", thread_id=thread_id, error=str(e)
        )
        return None


def get_state_snapshot(
    graph: CompiledStateGraph, thread_id: str
) -> Optional[StateSnapshot]:
    """
    Retrieve the state snapshot from the graph for a given thread_id.

    This is a synchronous wrapper around get_state_snapshot_async.
    Use this for backward compatibility in synchronous contexts.

    Args:
        graph: The compiled LangGraph state machine
        thread_id: The thread/conversation ID to retrieve state for

    Returns:
        StateSnapshot if found, None otherwise
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(get_state_snapshot_async(graph, thread_id))
    except Exception as e:
        logger.warning("Error in synchronous state snapshot retrieval", error=str(e))
        return None


def get_genie_conversation_ids_from_state(
    state_snapshot: Optional[StateSnapshot],
) -> dict[str, str]:
    """
    Extract genie_conversation_ids from a state snapshot.

    This function extracts the genie_conversation_ids dictionary from the state
    snapshot values if present.

    Args:
        state_snapshot: The state snapshot to extract conversation IDs from

    Returns:
        A dictionary mapping genie space_id to conversation_id, or empty dict if not found
    """
    if state_snapshot is None:
        return {}

    try:
        # Extract state values - these contain the actual state data
        state_values: dict[str, Any] = state_snapshot.values

        # Extract genie_conversation_ids from state values
        genie_conversation_ids: dict[str, str] = state_values.get(
            "genie_conversation_ids", {}
        )

        if genie_conversation_ids:
            logger.trace(
                "Retrieved genie conversation IDs", count=len(genie_conversation_ids)
            )
            return genie_conversation_ids

        return {}

    except Exception as e:
        logger.warning(
            "Error extracting genie conversation IDs from state", error=str(e)
        )
        return {}


def _extract_interrupt_value(interrupt: Interrupt) -> HITLRequest:
    """
    Extract the HITL request from a LangGraph Interrupt object.

    Following LangChain patterns, the Interrupt object has a .value attribute
    containing the HITLRequest structure with action_requests and review_configs.

    Args:
        interrupt: Interrupt object from LangGraph with .value and .id attributes

    Returns:
        HITLRequest with action_requests and review_configs
    """
    # Interrupt.value is typed as Any, but for HITL it should be a HITLRequest dict
    if isinstance(interrupt.value, dict):
        # Return as HITLRequest TypedDict
        return interrupt.value  # type: ignore[return-value]

    # Fallback: return empty structure if value is not a dict
    return {"action_requests": [], "review_configs": []}


def _format_action_requests_message(interrupt_data: list[HITLRequest]) -> str:
    """
    Format action requests from interrupts into a simple, user-friendly message.

    Since we now use LLM-based parsing, users can respond in natural language.
    This function just shows WHAT actions are pending, not HOW to respond.

    Args:
        interrupt_data: List of HITLRequest structures containing action_requests and review_configs

    Returns:
        Simple formatted message describing the pending actions
    """
    if not interrupt_data:
        return ""

    # Collect all action requests and review configs from all interrupts
    all_actions: list[ActionRequest] = []
    review_configs_map: dict[str, ReviewConfig] = {}

    for hitl_request in interrupt_data:
        all_actions.extend(hitl_request.get("action_requests", []))
        for review_config in hitl_request.get("review_configs", []):
            action_name = review_config.get("action_name", "")
            if action_name:
                review_configs_map[action_name] = review_config

    if not all_actions:
        return ""

    # Build simple, clean message
    lines = ["⏸️ **Action Approval Required**", ""]
    lines.append(
        f"The assistant wants to perform {len(all_actions)} action(s) that require your approval:"
    )
    lines.append("")

    for i, action in enumerate(all_actions, 1):
        tool_name = action.get("name", "unknown")
        args = action.get("args", {})
        description = action.get("description")

        lines.append(f"**{i}. {tool_name}**")

        # Show review prompt/description if available
        if description:
            lines.append(f"   • **Review:** {description}")

        if args:
            # Format args nicely, truncating long values
            for key, value in args.items():
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                lines.append(f"   • {key}: `{value_str}`")
        else:
            lines.append("   • (no arguments)")

        # Show allowed decisions
        review_config = review_configs_map.get(tool_name)
        if review_config:
            allowed_decisions = review_config.get("allowed_decisions", [])
            if allowed_decisions:
                decisions_str = ", ".join(allowed_decisions)
                lines.append(f"   • **Options:** {decisions_str}")

        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "**You can respond in natural language** (e.g., 'approve both', 'reject the first one', "
        "'change the email to new@example.com')"
    )
    lines.append("")
    lines.append(
        "Or provide structured decisions in `custom_inputs` with key `decisions`: "
        '`[{"type": "approve"}, {"type": "reject", "message": "reason"}]`'
    )

    return "\n".join(lines)


class LanggraphChatModel(ChatModel):
    """
    ChatModel that delegates requests to a LangGraph CompiledStateGraph.
    """

    def __init__(self, graph: CompiledStateGraph) -> None:
        self.graph = graph

    def predict(
        self, context, messages: list[ChatMessage], params: Optional[ChatParams] = None
    ) -> ChatCompletionResponse:
        logger.trace(
            "Predict called",
            messages_count=len(messages),
            has_params=params is not None,
        )
        if not messages:
            raise ValueError("Message list is empty.")

        request = {"messages": self._convert_messages_to_dict(messages)}

        context: Context = self._convert_to_context(params)
        custom_inputs: dict[str, Any] = {"configurable": context.model_dump()}

        # Use async ainvoke internally for parallel execution
        import asyncio

        async def _async_invoke():
            return await self.graph.ainvoke(
                request, context=context, config=custom_inputs
            )

        loop = asyncio.get_event_loop()
        response: dict[str, Sequence[BaseMessage]] = loop.run_until_complete(
            _async_invoke()
        )

        logger.trace(
            "Predict response received",
            messages_count=len(response.get("messages", [])),
        )

        last_message: BaseMessage = response["messages"][-1]

        response_message = ChatMessage(role="assistant", content=last_message.content)
        return ChatCompletionResponse(choices=[ChatChoice(message=response_message)])

    def _convert_to_context(
        self, params: Optional[ChatParams | dict[str, Any]]
    ) -> Context:
        input_data = params
        if isinstance(params, ChatParams):
            input_data = params.to_dict()

        configurable: dict[str, Any] = {}
        if "configurable" in input_data:
            configurable = input_data.pop("configurable")
        if "custom_inputs" in input_data:
            custom_inputs: dict[str, Any] = input_data.pop("custom_inputs")
            if "configurable" in custom_inputs:
                configurable = custom_inputs.pop("configurable")

        # Extract known Context fields
        user_id: str | None = configurable.pop("user_id", None)
        if user_id:
            user_id = user_id.replace(".", "_")

        # Accept either thread_id or conversation_id (interchangeable)
        # conversation_id takes precedence (Databricks vocabulary)
        thread_id: str | None = configurable.pop("thread_id", None)
        conversation_id: str | None = configurable.pop("conversation_id", None)

        # conversation_id takes precedence if both provided
        if conversation_id:
            thread_id = conversation_id
        if not thread_id:
            thread_id = str(uuid.uuid4())

        # All remaining configurable values become top-level context attributes
        return Context(
            user_id=user_id,
            thread_id=thread_id,
            **configurable,  # Extra fields become top-level attributes
        )

    def predict_stream(
        self, context, messages: list[ChatMessage], params: ChatParams
    ) -> Generator[ChatCompletionChunk, None, None]:
        logger.trace(
            "Predict stream called",
            messages_count=len(messages),
            has_params=params is not None,
        )
        if not messages:
            raise ValueError("Message list is empty.")

        request = {"messages": self._convert_messages_to_dict(messages)}

        context: Context = self._convert_to_context(params)
        custom_inputs: dict[str, Any] = {"configurable": context.model_dump()}

        # Use async astream internally for parallel execution
        import asyncio

        async def _async_stream():
            async for nodes, stream_mode, messages_batch in self.graph.astream(
                request,
                context=context,
                config=custom_inputs,
                stream_mode=["messages", "custom"],
                subgraphs=True,
            ):
                nodes: tuple[str, ...]
                stream_mode: str
                messages_batch: Sequence[BaseMessage]
                logger.trace(
                    "Stream batch received",
                    nodes=nodes,
                    stream_mode=stream_mode,
                    messages_count=len(messages_batch),
                )
                for message in messages_batch:
                    if (
                        isinstance(
                            message,
                            (
                                AIMessageChunk,
                                AIMessage,
                            ),
                        )
                        and message.content
                        and "summarization" not in nodes
                    ):
                        content = message.content
                        yield self._create_chat_completion_chunk(content)

        # Convert async generator to sync generator
        loop = asyncio.get_event_loop()
        async_gen = _async_stream()

        try:
            while True:
                try:
                    item = loop.run_until_complete(async_gen.__anext__())
                    yield item
                except StopAsyncIteration:
                    break
        finally:
            loop.run_until_complete(async_gen.aclose())

    def _create_chat_completion_chunk(self, content: str) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            choices=[
                ChatChunkChoice(
                    delta=ChatChoiceDelta(role="assistant", content=content)
                )
            ]
        )

    def _convert_messages_to_dict(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, Any]]:
        return [m.to_dict() for m in messages]


def _create_decision_schema(interrupt_data: list[HITLRequest]) -> type[BaseModel]:
    """
    Dynamically create a Pydantic model for structured output based on interrupt actions.

    This creates a schema that matches the expected decision format for the interrupted actions.
    Each action gets a corresponding decision field that can be approve, edit, or reject.
    Includes validation fields to ensure the response is complete and valid.

    Args:
        interrupt_data: List of HITL interrupt requests containing action_requests and review_configs

    Returns:
        A dynamically created Pydantic BaseModel class for structured output

    Example:
        For two actions (send_email, execute_sql), creates a model like:
        class Decisions(BaseModel):
            is_valid: bool
            validation_message: Optional[str]
            decision_1: Literal["approve", "edit", "reject"]
            decision_1_message: Optional[str]  # For reject
            decision_1_edited_args: Optional[dict]  # For edit
            decision_2: Literal["approve", "edit", "reject"]
            ...
    """
    # Collect all actions
    all_actions: list[ActionRequest] = []
    review_configs_map: dict[str, ReviewConfig] = {}

    for hitl_request in interrupt_data:
        all_actions.extend(hitl_request.get("action_requests", []))
        review_config: ReviewConfig
        for review_config in hitl_request.get("review_configs", []):
            action_name: str = review_config.get("action_name", "")
            if action_name:
                review_configs_map[action_name] = review_config

    # Build fields for the dynamic model
    # Start with validation fields
    fields: dict[str, Any] = {
        "is_valid": (
            bool,
            Field(
                description="Whether the user's response provides valid decisions for ALL actions. "
                "Set to False if the user's message is unclear, ambiguous, or doesn't provide decisions for all actions."
            ),
        ),
        "validation_message": (
            Optional[str],
            Field(
                None,
                description="If is_valid is False, explain what is missing or unclear. "
                "Be specific about which action(s) need clarification.",
            ),
        ),
    }

    i: int
    action: ActionRequest
    for i, action in enumerate(all_actions, 1):
        tool_name: str = action.get("name", "unknown")
        review_config: Optional[ReviewConfig] = review_configs_map.get(tool_name)
        allowed_decisions: list[str] = (
            review_config.get("allowed_decisions", ["approve", "reject"])
            if review_config
            else ["approve", "reject"]
        )

        # Create a Literal type for allowed decisions
        decision_literal: type = Literal[tuple(allowed_decisions)]  # type: ignore

        # Add decision field
        fields[f"decision_{i}"] = (
            decision_literal,
            Field(
                description=f"Decision for action {i} ({tool_name}): {', '.join(allowed_decisions)}"
            ),
        )

        # Add optional message field for reject
        if "reject" in allowed_decisions:
            fields[f"decision_{i}_message"] = (
                Optional[str],
                Field(
                    None,
                    description=f"Optional message if rejecting action {i}",
                ),
            )

        # Add optional edited_args field for edit
        if "edit" in allowed_decisions:
            fields[f"decision_{i}_edited_args"] = (
                Optional[dict[str, Any]],
                Field(
                    None,
                    description=f"Modified arguments if editing action {i}. Only provide fields that need to change.",
                ),
            )

    # Create the dynamic model
    DecisionsModel = create_model(
        "InterruptDecisions",
        __doc__="Decisions for each interrupted action, in order.",
        **fields,
    )

    return DecisionsModel


def _convert_schema_to_decisions(
    parsed_output: BaseModel,
    interrupt_data: list[HITLRequest],
) -> list[Decision]:
    """
    Convert the parsed structured output into LangChain Decision objects.

    Args:
        parsed_output: The Pydantic model instance from structured output
        interrupt_data: Original interrupt data for context

    Returns:
        List of Decision dictionaries compatible with Command(resume={"decisions": ...})
    """
    # Collect all actions to know how many decisions we need
    all_actions: list[ActionRequest] = []
    hitl_request: HITLRequest
    for hitl_request in interrupt_data:
        all_actions.extend(hitl_request.get("action_requests", []))

    decisions: list[Decision] = []

    i: int
    for i in range(1, len(all_actions) + 1):
        decision_type: str = getattr(parsed_output, f"decision_{i}")

        if decision_type == "approve":
            decisions.append({"type": "approve"})  # type: ignore
        elif decision_type == "reject":
            message: Optional[str] = getattr(
                parsed_output, f"decision_{i}_message", None
            )
            reject_decision: RejectDecision = {"type": "reject"}
            if message:
                reject_decision["message"] = message
            decisions.append(reject_decision)  # type: ignore
        elif decision_type == "edit":
            edited_args: Optional[dict[str, Any]] = getattr(
                parsed_output, f"decision_{i}_edited_args", None
            )
            action: ActionRequest = all_actions[i - 1]
            tool_name: str = action.get("name", "")
            original_args: dict[str, Any] = action.get("args", {})

            # Merge original args with edited args
            final_args: dict[str, Any] = {**original_args, **(edited_args or {})}

            edit_decision: EditDecision = {
                "type": "edit",
                "edited_action": {
                    "name": tool_name,
                    "args": final_args,
                },
            }
            decisions.append(edit_decision)  # type: ignore

    return decisions


def handle_interrupt_response(
    snapshot: StateSnapshot,
    messages: list[BaseMessage],
    model: Optional[LanguageModelLike] = None,
) -> dict[str, Any]:
    """
    Parse user's natural language response to interrupts using LLM with structured output.

    This function uses an LLM to understand the user's intent and extract structured decisions
    for each pending action. The schema is dynamically created based on the pending actions.
    Includes validation to ensure the response is complete and valid.

    Args:
        snapshot: The current state snapshot containing interrupts
        messages: List of messages, from which the last human message will be extracted
        model: Optional LLM to use for parsing. Defaults to Llama 3.1 70B

    Returns:
        Dictionary with:
        - "is_valid": bool indicating if the response is valid
        - "validation_message": Optional message if invalid, explaining what's missing
        - "decisions": list of Decision objects (empty if invalid)

    Example:
        Valid: {"is_valid": True, "validation_message": None, "decisions": [{"type": "approve"}]}
        Invalid: {"is_valid": False, "validation_message": "Please specify...", "decisions": []}
    """
    # Extract the last human message
    user_message_obj: Optional[HumanMessage] = last_human_message(messages)

    if not user_message_obj:
        logger.warning("HITL: No human message found in interrupt response")
        return {
            "is_valid": False,
            "validation_message": "No user message found. Please provide a response to the pending action(s).",
            "decisions": [],
        }

    user_message: str = str(user_message_obj.content)
    logger.info(
        "HITL: Parsing user interrupt response", message_preview=user_message[:100]
    )

    if not model:
        model = ChatDatabricks(
            endpoint="databricks-claude-sonnet-4",
            temperature=0,
        )

    # Extract interrupt data
    if not snapshot.interrupts:
        logger.warning("HITL: No interrupts found in snapshot")
        return {"decisions": []}

    interrupt_data: list[HITLRequest] = [
        _extract_interrupt_value(interrupt) for interrupt in snapshot.interrupts
    ]

    # Collect all actions for context
    all_actions: list[ActionRequest] = []
    hitl_request: HITLRequest
    for hitl_request in interrupt_data:
        all_actions.extend(hitl_request.get("action_requests", []))

    if not all_actions:
        logger.warning("HITL: No actions found in interrupts")
        return {"decisions": []}

    # Create dynamic schema
    DecisionsModel: type[BaseModel] = _create_decision_schema(interrupt_data)

    # Create structured LLM
    structured_llm: LanguageModelLike = model.with_structured_output(DecisionsModel)

    # Format action context for the LLM
    action_descriptions: list[str] = []
    i: int
    action: ActionRequest
    for i, action in enumerate(all_actions, 1):
        tool_name: str = action.get("name", "unknown")
        args: dict[str, Any] = action.get("args", {})
        args_str: str = (
            ", ".join(f"{k}={v}" for k, v in args.items()) if args else "(no args)"
        )
        action_descriptions.append(f"Action {i}: {tool_name}({args_str})")

    system_prompt: str = f"""You are parsing a user's response to interrupted agent actions.

The following actions are pending approval:
{chr(10).join(action_descriptions)}

Your task is to extract the user's decision for EACH action in order. The user may:
- Approve: Accept the action as-is
- Reject: Cancel the action (optionally with a reason/message)
- Edit: Modify the arguments before executing

VALIDATION:
- Set is_valid=True only if you can confidently extract decisions for ALL actions
- Set is_valid=False if the user's message is:
  * Unclear or ambiguous
  * Missing decisions for some actions
  * Asking a question instead of providing decisions
  * Not addressing the actions at all
- If is_valid=False, provide a clear validation_message explaining what is needed

FLEXIBILITY:
- Be flexible in parsing informal language like "yes", "no", "ok", "change X to Y"
- If the user doesn't explicitly mention an action, assume they want to approve it
- Only mark as invalid if the message is genuinely unclear or incomplete"""

    try:
        # Invoke LLM with structured output
        parsed: BaseModel = structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
        )

        # Check validation first
        is_valid: bool = getattr(parsed, "is_valid", True)
        validation_message: Optional[str] = getattr(parsed, "validation_message", None)

        if not is_valid:
            logger.warning(
                "HITL: Invalid user response", reason=validation_message or "Unknown"
            )
            return {
                "is_valid": False,
                "validation_message": validation_message
                or "Your response was unclear. Please provide a clear decision for each action.",
                "decisions": [],
            }

        # Convert to Decision format
        decisions: list[Decision] = _convert_schema_to_decisions(parsed, interrupt_data)

        logger.info("HITL: Parsed interrupt decisions", decisions_count=len(decisions))
        return {"is_valid": True, "validation_message": None, "decisions": decisions}

    except Exception as e:
        logger.error("HITL: Failed to parse interrupt response", error=str(e))
        # Return invalid response on parsing failure
        return {
            "is_valid": False,
            "validation_message": f"Failed to parse your response: {str(e)}. Please provide a clear decision for each action.",
            "decisions": [],
        }


class LanggraphResponsesAgent(ResponsesAgent):
    """
    ResponsesAgent that delegates requests to a LangGraph CompiledStateGraph.

    This is the modern replacement for LanggraphChatModel, providing better
    support for streaming, tool calling, and async execution.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
    ) -> None:
        self.graph = graph

    async def apredict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Async version of predict - primary implementation for Databricks Apps.

        Process a ResponsesAgentRequest and return a ResponsesAgentResponse.
        This method can be awaited directly in async contexts (e.g., MLflow AgentServer).

        Input structure (custom_inputs):
            configurable:
                thread_id: "abc-123"        # Or conversation_id (aliases)
                user_id: "nate.fleming"
                store_num: "87887"
            session:  # Paste from previous output
                conversation_id: "abc-123"  # Alias of thread_id
                genie:
                    spaces:
                        space_123: {conversation_id: "conv_456", ...}
            decisions:  # For resuming interrupted graphs (HITL)
                - type: "approve"
                - type: "reject"
                  message: "Not authorized"

        Output structure (custom_outputs):
            configurable:
                thread_id: "abc-123"
                user_id: "nate.fleming"
                store_num: "87887"
            session:
                conversation_id: "abc-123"
                genie:
                    spaces:
                        space_123: {conversation_id: "conv_456", ...}
            pending_actions:  # If HITL interrupt occurred
                - name: "send_email"
                  arguments: {...}
                  description: "..."
        """
        from langgraph.types import Command

        # Extract conversation_id for logging
        conversation_id_for_log: str | None = None
        if request.context and hasattr(request.context, "conversation_id"):
            conversation_id_for_log = request.context.conversation_id
        elif request.custom_inputs:
            if "configurable" in request.custom_inputs and isinstance(
                request.custom_inputs["configurable"], dict
            ):
                conversation_id_for_log = request.custom_inputs["configurable"].get(
                    "conversation_id"
                )
            if (
                conversation_id_for_log is None
                and "session" in request.custom_inputs
                and isinstance(request.custom_inputs["session"], dict)
            ):
                conversation_id_for_log = request.custom_inputs["session"].get(
                    "conversation_id"
                )

        logger.debug(
            "ResponsesAgent apredict called",
            conversation_id=conversation_id_for_log
            if conversation_id_for_log
            else "new",
        )

        # Convert ResponsesAgent input to LangChain messages
        messages: list[dict[str, Any]] = self._convert_request_to_langchain_messages(
            request
        )

        # Prepare context (conversation_id -> thread_id mapping happens here)
        context: Context = self._convert_request_to_context(request)
        custom_inputs: dict[str, Any] = {"configurable": context.model_dump()}

        # Extract session state from request
        session_input: dict[str, Any] = self._extract_session_from_request(request)

        try:
            # Check if this is a resume request (HITL)
            if request.custom_inputs and "decisions" in request.custom_inputs:
                # Explicit structured decisions
                decisions: list[Decision] = request.custom_inputs["decisions"]
                logger.info(
                    "HITL: Resuming with explicit decisions",
                    decisions_count=len(decisions),
                )

                # Resume interrupted graph with decisions
                response = await self.graph.ainvoke(
                    Command(resume={"decisions": decisions}),
                    context=context,
                    config=custom_inputs,
                )
            elif self.graph.checkpointer:
                # Check if graph is currently interrupted
                snapshot: StateSnapshot = await self.graph.aget_state(
                    config=custom_inputs
                )
                if is_interrupted(snapshot):
                    logger.info("HITL: Graph interrupted, checking for user response")

                    # Convert message dicts to BaseMessage objects
                    message_objects: list[BaseMessage] = convert_openai_messages(
                        messages
                    )

                    # Parse user's message with LLM to extract decisions
                    parsed_result: dict[str, Any] = handle_interrupt_response(
                        snapshot=snapshot,
                        messages=message_objects,
                        model=None,
                    )

                    if not parsed_result.get("is_valid", False):
                        validation_message: str = parsed_result.get(
                            "validation_message",
                            "Your response was unclear. Please provide a clear decision for each action.",
                        )
                        logger.warning(
                            "HITL: Invalid response from user",
                            validation_message=validation_message,
                        )

                        # Return error message without resuming
                        response = {
                            "messages": [
                                AIMessage(
                                    content=f"❌ **Invalid Response**\n\n{validation_message}"
                                )
                            ]
                        }
                    else:
                        decisions = parsed_result.get("decisions", [])
                        logger.info(
                            "HITL: LLM parsed valid decisions from user message",
                            decisions_count=len(decisions),
                        )

                        # Resume interrupted graph with parsed decisions
                        response = await self.graph.ainvoke(
                            Command(resume={"decisions": decisions}),
                            context=context,
                            config=custom_inputs,
                        )
                else:
                    # Normal invocation
                    graph_input: dict[str, Any] = {"messages": messages}
                    if "genie_conversation_ids" in session_input:
                        graph_input["genie_conversation_ids"] = session_input[
                            "genie_conversation_ids"
                        ]

                    response = await self.graph.ainvoke(
                        graph_input, context=context, config=custom_inputs
                    )
            else:
                # No checkpointer, use normal invocation
                graph_input = {"messages": messages}
                if "genie_conversation_ids" in session_input:
                    graph_input["genie_conversation_ids"] = session_input[
                        "genie_conversation_ids"
                    ]

                response = await self.graph.ainvoke(
                    graph_input, context=context, config=custom_inputs
                )
        except Exception as e:
            logger.error("Error in graph invocation", error=str(e))
            raise

        # Convert response to ResponsesAgent format
        last_message: BaseMessage = response["messages"][-1]

        # Build custom_outputs
        custom_outputs: dict[str, Any] = await self._build_custom_outputs_async(
            context=context,
            thread_id=context.thread_id,
        )

        # Handle structured_response if present
        if "structured_response" in response:
            from dataclasses import asdict, is_dataclass

            from pydantic import BaseModel

            structured_response = response["structured_response"]
            logger.trace(
                "Processing structured response",
                response_type=type(structured_response).__name__,
            )

            if isinstance(structured_response, BaseModel):
                serialized: dict[str, Any] = structured_response.model_dump()
            elif is_dataclass(structured_response):
                serialized = asdict(structured_response)
            elif isinstance(structured_response, dict):
                serialized = structured_response
            else:
                serialized = (
                    dict(structured_response)
                    if hasattr(structured_response, "__dict__")
                    else structured_response
                )

            import json

            structured_text: str = json.dumps(serialized, indent=2)
            output_item = self.create_text_output_item(
                text=structured_text, id=f"msg_{uuid.uuid4().hex[:8]}"
            )
            logger.trace("Structured response placed in message content")
        else:
            output_item = self.create_text_output_item(
                text=last_message.content, id=f"msg_{uuid.uuid4().hex[:8]}"
            )

        # Include interrupt structure if HITL occurred
        if "__interrupt__" in response:
            interrupts: list[Interrupt] = response["__interrupt__"]
            logger.info("HITL: Interrupts detected", interrupts_count=len(interrupts))

            seen_interrupt_ids: set[str] = set()
            interrupt_data: list[HITLRequest] = []
            for interrupt in interrupts:
                if interrupt.id not in seen_interrupt_ids:
                    seen_interrupt_ids.add(interrupt.id)
                    interrupt_data.append(_extract_interrupt_value(interrupt))
                    logger.trace(
                        "HITL: Added interrupt to response", interrupt_id=interrupt.id
                    )

            custom_outputs["interrupts"] = interrupt_data
            logger.debug(
                "HITL: Included interrupts in response",
                interrupts_count=len(interrupt_data),
            )

            action_message: str = _format_action_requests_message(interrupt_data)
            if action_message:
                output_item = self.create_text_output_item(
                    text=action_message, id=f"msg_{uuid.uuid4().hex[:8]}"
                )

        return ResponsesAgentResponse(
            output=[output_item], custom_outputs=custom_outputs
        )

    async def apredict_stream(
        self, request: ResponsesAgentRequest
    ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
        """
        Async version of predict_stream - primary implementation for Databricks Apps.

        Process a ResponsesAgentRequest and yield ResponsesAgentStreamEvent objects.
        This method can be used directly with async for loops in async contexts.

        Uses same input/output structure as apredict() for consistency.
        Supports Human-in-the-Loop (HITL) interrupts.
        """
        from langgraph.types import Command

        # Extract conversation_id for logging
        conversation_id_for_log: str | None = None
        if request.context and hasattr(request.context, "conversation_id"):
            conversation_id_for_log = request.context.conversation_id
        elif request.custom_inputs:
            if "configurable" in request.custom_inputs and isinstance(
                request.custom_inputs["configurable"], dict
            ):
                conversation_id_for_log = request.custom_inputs["configurable"].get(
                    "conversation_id"
                )
            if (
                conversation_id_for_log is None
                and "session" in request.custom_inputs
                and isinstance(request.custom_inputs["session"], dict)
            ):
                conversation_id_for_log = request.custom_inputs["session"].get(
                    "conversation_id"
                )

        logger.debug(
            "ResponsesAgent apredict_stream called",
            conversation_id=conversation_id_for_log
            if conversation_id_for_log
            else "new",
        )

        # Convert ResponsesAgent input to LangChain messages
        messages: list[dict[str, Any]] = self._convert_request_to_langchain_messages(
            request
        )

        # Prepare context (conversation_id -> thread_id mapping happens here)
        context: Context = self._convert_request_to_context(request)
        custom_inputs: dict[str, Any] = {"configurable": context.model_dump()}

        # Extract session state from request
        session_input: dict[str, Any] = self._extract_session_from_request(request)

        item_id: str = f"msg_{uuid.uuid4().hex[:8]}"
        accumulated_content: str = ""
        interrupt_data: list[HITLRequest] = []
        seen_interrupt_ids: set[str] = set()
        structured_response: Any = None

        try:
            # Check if this is a resume request (HITL)
            if request.custom_inputs and "decisions" in request.custom_inputs:
                decisions: list[Decision] = request.custom_inputs["decisions"]
                logger.info(
                    "HITL: Resuming stream with explicit decisions",
                    decisions_count=len(decisions),
                )
                stream_input: Command | dict[str, Any] = Command(
                    resume={"decisions": decisions}
                )
            elif self.graph.checkpointer:
                snapshot: StateSnapshot = await self.graph.aget_state(
                    config=custom_inputs
                )
                if is_interrupted(snapshot):
                    logger.info(
                        "HITL: Graph interrupted, checking for user response in stream"
                    )

                    message_objects: list[BaseMessage] = convert_openai_messages(
                        messages
                    )

                    parsed_result: dict[str, Any] = handle_interrupt_response(
                        snapshot=snapshot,
                        messages=message_objects,
                        model=None,
                    )

                    if not parsed_result.get("is_valid", False):
                        validation_message: str = parsed_result.get(
                            "validation_message",
                            "Your response was unclear. Please provide a clear decision for each action.",
                        )
                        logger.warning(
                            "HITL: Invalid response from user in stream",
                            validation_message=validation_message,
                        )

                        custom_outputs: dict[
                            str, Any
                        ] = await self._build_custom_outputs_async(
                            context=context,
                            thread_id=context.thread_id,
                        )

                        error_message: str = (
                            f"❌ **Invalid Response**\n\n{validation_message}"
                        )
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=self.create_text_output_item(
                                text=error_message, id=item_id
                            ),
                            custom_outputs=custom_outputs,
                        )
                        return

                    decisions = parsed_result.get("decisions", [])
                    logger.info(
                        "HITL: LLM parsed valid decisions from user message in stream",
                        decisions_count=len(decisions),
                    )

                    stream_input = Command(resume={"decisions": decisions})
                else:
                    graph_input: dict[str, Any] = {"messages": messages}
                    if "genie_conversation_ids" in session_input:
                        graph_input["genie_conversation_ids"] = session_input[
                            "genie_conversation_ids"
                        ]
                    stream_input = graph_input
            else:
                graph_input = {"messages": messages}
                if "genie_conversation_ids" in session_input:
                    graph_input["genie_conversation_ids"] = session_input[
                        "genie_conversation_ids"
                    ]
                stream_input = graph_input

            # Stream the graph execution
            async for nodes, stream_mode, data in self.graph.astream(
                stream_input,
                context=context,
                config=custom_inputs,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                nodes: tuple[str, ...]
                stream_mode: str

                if stream_mode == "messages":
                    messages_batch: Sequence[BaseMessage] = data
                    for message in messages_batch:
                        if (
                            isinstance(message, (AIMessageChunk, AIMessage))
                            and message.content
                            and "summarization" not in nodes
                        ):
                            content: str = message.content
                            accumulated_content += content

                            yield ResponsesAgentStreamEvent(
                                **self.create_text_delta(delta=content, item_id=item_id)
                            )

                elif stream_mode == "updates":
                    updates: dict[str, Any] = data
                    for source, update in updates.items():
                        if source == "__interrupt__":
                            interrupts: list[Interrupt] = update
                            logger.info(
                                "HITL: Interrupts detected during streaming",
                                interrupts_count=len(interrupts),
                            )

                            for interrupt in interrupts:
                                if interrupt.id not in seen_interrupt_ids:
                                    seen_interrupt_ids.add(interrupt.id)
                                    interrupt_data.append(
                                        _extract_interrupt_value(interrupt)
                                    )
                                    logger.trace(
                                        "HITL: Added interrupt to response",
                                        interrupt_id=interrupt.id,
                                    )
                        elif (
                            isinstance(update, dict) and "structured_response" in update
                        ):
                            structured_response = update["structured_response"]
                            logger.trace(
                                "Captured structured response from stream",
                                response_type=type(structured_response).__name__,
                            )

            # Get final state if checkpointer available
            if self.graph.checkpointer:
                final_state: StateSnapshot = await self.graph.aget_state(
                    config=custom_inputs
                )
                if (
                    "structured_response" in final_state.values
                    and not structured_response
                ):
                    structured_response = final_state.values["structured_response"]

            # Build custom_outputs
            custom_outputs = await self._build_custom_outputs_async(
                context=context,
                thread_id=context.thread_id,
            )

            # Handle structured_response in streaming
            output_text: str = accumulated_content
            if structured_response:
                from dataclasses import asdict, is_dataclass

                from pydantic import BaseModel

                logger.trace(
                    "Processing structured response in streaming",
                    response_type=type(structured_response).__name__,
                )

                if isinstance(structured_response, BaseModel):
                    serialized: dict[str, Any] = structured_response.model_dump()
                elif is_dataclass(structured_response):
                    serialized = asdict(structured_response)
                elif isinstance(structured_response, dict):
                    serialized = structured_response
                else:
                    serialized = (
                        dict(structured_response)
                        if hasattr(structured_response, "__dict__")
                        else structured_response
                    )

                import json

                structured_text: str = json.dumps(serialized, indent=2)

                if accumulated_content.strip():
                    yield ResponsesAgentStreamEvent(
                        **self.create_text_delta(delta="\n\n", item_id=item_id)
                    )
                    yield ResponsesAgentStreamEvent(
                        **self.create_text_delta(delta=structured_text, item_id=item_id)
                    )
                    output_text = f"{accumulated_content}\n\n{structured_text}"
                else:
                    yield ResponsesAgentStreamEvent(
                        **self.create_text_delta(delta=structured_text, item_id=item_id)
                    )
                    output_text = structured_text

                logger.trace("Streamed structured response in message content")

            # Include interrupt structure if HITL occurred
            if interrupt_data:
                custom_outputs["interrupts"] = interrupt_data
                logger.info(
                    "HITL: Included interrupts in streaming response",
                    interrupts_count=len(interrupt_data),
                )

                action_message = _format_action_requests_message(interrupt_data)
                if action_message:
                    if not accumulated_content:
                        output_text = action_message
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(
                                delta=action_message, item_id=item_id
                            )
                        )
                    else:
                        output_text = f"{accumulated_content}\n\n{action_message}"
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta="\n\n", item_id=item_id)
                        )
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(
                                delta=action_message, item_id=item_id
                            )
                        )

            # Yield final output item
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(text=output_text, id=item_id),
                custom_outputs=custom_outputs,
            )
        except Exception as e:
            logger.error("Error in graph streaming", error=str(e))
            raise

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Synchronous wrapper for apredict().

        Process a ResponsesAgentRequest and return a ResponsesAgentResponse.
        For async contexts (e.g., Databricks Apps), use apredict() directly.

        Note: This method uses asyncio.run() internally, which will fail in contexts
        where an event loop is already running (e.g., uvloop). For those cases,
        use apredict() instead.
        """
        import asyncio

        logger.debug("ResponsesAgent predict called (sync wrapper)")
        return asyncio.run(self.apredict(request))

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Synchronous wrapper for apredict_stream().

        Process a ResponsesAgentRequest and yield ResponsesAgentStreamEvent objects.
        For async contexts (e.g., Databricks Apps), use apredict_stream() directly.

        Note: This method converts the async generator to a sync generator using
        event loop manipulation. For contexts where an event loop is already running
        (e.g., uvloop), use apredict_stream() instead.
        """
        import asyncio

        logger.debug("ResponsesAgent predict_stream called (sync wrapper)")

        # Convert async generator to sync generator
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async_gen = self.apredict_stream(request)

        try:
            while True:
                try:
                    item = loop.run_until_complete(async_gen.__anext__())
                    yield item
                except StopAsyncIteration:
                    break
                except Exception as e:
                    logger.error("Error in streaming", error=str(e))
                    raise
        finally:
            try:
                loop.run_until_complete(async_gen.aclose())
            except Exception as e:
                logger.warning("Error closing async generator", error=str(e))

    def _extract_text_from_content(
        self,
        content: Union[str, list[Union[ResponseInputTextParam, str, dict[str, Any]]]],
    ) -> str:
        """Extract text content from various MLflow content formats.

        MLflow ResponsesAgent supports multiple content formats:
        - str: Simple text content
        - list[ResponseInputTextParam]: Structured text objects with .text attribute
        - list[dict]: Dictionaries with "text" key
        - Mixed lists of the above types

        This method normalizes all formats to a single concatenated string.

        Args:
            content: The content to extract text from

        Returns:
            Concatenated text string from all content items
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for content_item in content:
                if isinstance(content_item, ResponseInputTextParam):
                    text_parts.append(content_item.text)
                elif isinstance(content_item, str):
                    text_parts.append(content_item)
                elif isinstance(content_item, dict) and "text" in content_item:
                    text_parts.append(content_item["text"])
            return "".join(text_parts)
        else:
            # Fallback for unknown types - try to extract text attribute
            return getattr(content, "text", str(content))

    def _convert_request_to_langchain_messages(
        self, request: ResponsesAgentRequest
    ) -> list[dict[str, Any]]:
        """Convert ResponsesAgent input to LangChain message format."""
        messages = []

        for input_item in request.input:
            if isinstance(input_item, Message):
                # Handle MLflow Message objects
                content = self._extract_text_from_content(input_item.content)
                messages.append({"role": input_item.role, "content": content})
            elif isinstance(input_item, dict):
                # Handle dict format
                if "role" in input_item and "content" in input_item:
                    content = self._extract_text_from_content(input_item["content"])
                    messages.append({"role": input_item["role"], "content": content})
            else:
                # Fallback for other object types with role/content attributes
                role = getattr(input_item, "role", "user")
                content = self._extract_text_from_content(
                    getattr(input_item, "content", "")
                )
                messages.append({"role": role, "content": content})

        return messages

    def _convert_request_to_context(self, request: ResponsesAgentRequest) -> Context:
        """Convert ResponsesAgent context to internal Context.

        Handles the input structure:
        - custom_inputs.configurable: Configuration (thread_id, user_id, store_num, etc.)
        - custom_inputs.session: Accumulated state (conversation_id, genie conversations, etc.)

        Maps conversation_id -> thread_id for LangGraph compatibility.
        conversation_id can be provided in either configurable or session.
        Normalizes user_id (replaces . with _) for memory namespace compatibility.
        """
        logger.trace(
            "Converting request to context",
            has_context=request.context is not None,
            has_custom_inputs=request.custom_inputs is not None,
        )

        configurable: dict[str, Any] = {}
        session: dict[str, Any] = {}

        # Process context values first (lower priority)
        # These come from Databricks ResponsesAgent ChatContext
        chat_context: Optional[ChatContext] = request.context
        if chat_context is not None:
            conversation_id: Optional[str] = chat_context.conversation_id
            user_id: Optional[str] = chat_context.user_id

            if conversation_id is not None:
                configurable["conversation_id"] = conversation_id

            if user_id is not None:
                configurable["user_id"] = user_id

        # Process custom_inputs after context so they can override context values (higher priority)
        if request.custom_inputs:
            # Extract configurable section (user config)
            if "configurable" in request.custom_inputs:
                configurable.update(request.custom_inputs["configurable"])

            # Extract session section
            if "session" in request.custom_inputs:
                session_input = request.custom_inputs["session"]
                if isinstance(session_input, dict):
                    session = session_input

            # Handle legacy flat structure (backwards compatibility)
            # If user passes keys directly in custom_inputs, merge them
            for key in list(request.custom_inputs.keys()):
                if key not in ("configurable", "session"):
                    configurable[key] = request.custom_inputs[key]

        # Extract known Context fields
        user_id_value: str | None = configurable.pop("user_id", None)
        if user_id_value:
            # Normalize user_id for memory namespace (replace . with _)
            user_id_value = user_id_value.replace(".", "_")

        # Accept thread_id from configurable, or conversation_id from configurable or session
        # Priority: configurable.conversation_id > session.conversation_id > configurable.thread_id
        thread_id: str | None = configurable.pop("thread_id", None)
        conversation_id: str | None = configurable.pop("conversation_id", None)

        # Also check session for conversation_id (output puts it there)
        if conversation_id is None and "conversation_id" in session:
            conversation_id = session.get("conversation_id")

        # conversation_id takes precedence if provided
        if conversation_id:
            thread_id = conversation_id
        if not thread_id:
            # Generate new thread_id if neither provided
            thread_id = str(uuid.uuid4())

        # All remaining configurable values become top-level context attributes
        logger.trace(
            "Creating context",
            user_id=user_id_value,
            thread_id=thread_id,
            extra_keys=list(configurable.keys()) if configurable else [],
        )

        return Context(
            user_id=user_id_value,
            thread_id=thread_id,
            **configurable,  # Pass remaining configurable values as context attributes
        )

    def _extract_session_from_request(
        self, request: ResponsesAgentRequest
    ) -> dict[str, Any]:
        """Extract session state from request for passing to graph.

        Handles:
        - New structure: custom_inputs.session.genie
        - Legacy structure: custom_inputs.genie_conversation_ids
        """
        session: dict[str, Any] = {}

        if not request.custom_inputs:
            return session

        # New structure: session.genie
        if "session" in request.custom_inputs:
            session_input = request.custom_inputs["session"]
            if isinstance(session_input, dict) and "genie" in session_input:
                genie_state = session_input["genie"]
                # Extract conversation IDs from the new structure
                if isinstance(genie_state, dict) and "spaces" in genie_state:
                    genie_conversation_ids = {}
                    for space_id, space_state in genie_state["spaces"].items():
                        if (
                            isinstance(space_state, dict)
                            and "conversation_id" in space_state
                        ):
                            genie_conversation_ids[space_id] = space_state[
                                "conversation_id"
                            ]
                    if genie_conversation_ids:
                        session["genie_conversation_ids"] = genie_conversation_ids

        # Legacy structure: genie_conversation_ids at top level
        if "genie_conversation_ids" in request.custom_inputs:
            session["genie_conversation_ids"] = request.custom_inputs[
                "genie_conversation_ids"
            ]

        # Also check inside configurable for legacy support
        if "configurable" in request.custom_inputs:
            cfg = request.custom_inputs["configurable"]
            if isinstance(cfg, dict) and "genie_conversation_ids" in cfg:
                session["genie_conversation_ids"] = cfg["genie_conversation_ids"]

        return session

    def _build_custom_outputs(
        self,
        context: Context,
        thread_id: Optional[str],
        loop: Any,  # asyncio.AbstractEventLoop
    ) -> dict[str, Any]:
        """Build custom_outputs that can be copy-pasted as next request's custom_inputs.

        Output structure:
            configurable:
                thread_id: "abc-123"        # Thread identifier (conversation_id is alias)
                user_id: "nate.fleming"     # De-normalized (no underscore replacement)
                store_num: "87887"          # Any custom fields
            session:
                conversation_id: "abc-123"  # Alias of thread_id for Databricks compatibility
                genie:
                    spaces:
                        space_123: {conversation_id: "conv_456", cache_hit: false, ...}
        """
        return loop.run_until_complete(
            self._build_custom_outputs_async(context=context, thread_id=thread_id)
        )

    async def _build_custom_outputs_async(
        self,
        context: Context,
        thread_id: Optional[str],
    ) -> dict[str, Any]:
        """Async version of _build_custom_outputs."""
        # Build configurable section
        # Note: only thread_id is included here (conversation_id goes in session)
        configurable: dict[str, Any] = {}

        if thread_id:
            configurable["thread_id"] = thread_id

        # Include user_id (keep normalized form for consistency)
        if context.user_id:
            configurable["user_id"] = context.user_id

        # Include all extra fields from context (beyond user_id and thread_id)
        context_dict = context.model_dump()
        for key, value in context_dict.items():
            if key not in {"user_id", "thread_id"} and value is not None:
                configurable[key] = value

        # Build session section with accumulated state
        # Note: conversation_id is included here as an alias of thread_id
        session: dict[str, Any] = {}

        if thread_id:
            # Include conversation_id in session (alias of thread_id)
            session["conversation_id"] = thread_id

            state_snapshot: Optional[StateSnapshot] = await get_state_snapshot_async(
                self.graph, thread_id
            )
            genie_conversation_ids: dict[str, str] = (
                get_genie_conversation_ids_from_state(state_snapshot)
            )
            if genie_conversation_ids:
                # Convert flat genie_conversation_ids to new session.genie.spaces structure
                session["genie"] = {
                    "spaces": {
                        space_id: {
                            "conversation_id": conv_id,
                            # Note: cache_hit, follow_up_questions populated by Genie tool
                            "cache_hit": False,
                            "follow_up_questions": [],
                        }
                        for space_id, conv_id in genie_conversation_ids.items()
                    }
                }

        return {
            "configurable": configurable,
            "session": session,
        }


def create_agent(graph: CompiledStateGraph) -> ChatAgent:
    """
    Create an MLflow-compatible ChatAgent from a LangGraph state machine.

    Factory function that wraps a compiled LangGraph in the LangGraphChatAgent
    class to make it deployable through MLflow.

    Args:
        graph: A compiled LangGraph state machine

    Returns:
        An MLflow-compatible ChatAgent instance
    """
    return LanggraphChatModel(graph)


def create_responses_agent(
    graph: CompiledStateGraph,
) -> ResponsesAgent:
    """
    Create an MLflow-compatible ResponsesAgent from a LangGraph state machine.

    Factory function that wraps a compiled LangGraph in the LanggraphResponsesAgent
    class to make it deployable through MLflow.

    Args:
        graph: A compiled LangGraph state machine

    Returns:
        An MLflow-compatible ResponsesAgent instance
    """
    return LanggraphResponsesAgent(graph)


def _process_langchain_messages(
    app: LanggraphChatModel | CompiledStateGraph,
    messages: Sequence[BaseMessage],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> dict[str, Any] | Any:
    """Process LangChain messages using async LangGraph calls internally."""
    import asyncio

    if isinstance(app, LanggraphChatModel):
        app = app.graph

    # Use async ainvoke internally for parallel execution
    async def _async_invoke():
        return await app.ainvoke({"messages": messages}, config=custom_inputs)

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_async_invoke())


def _configurable_to_context(configurable: dict[str, Any]) -> Context:
    """Convert a configurable dict to a Context object."""
    configurable = configurable.copy()

    # Extract known Context fields
    user_id: str | None = configurable.pop("user_id", None)
    if user_id:
        user_id = user_id.replace(".", "_")

    thread_id: str | None = configurable.pop("thread_id", None)
    if "conversation_id" in configurable and not thread_id:
        thread_id = configurable.pop("conversation_id")
    if not thread_id:
        thread_id = str(uuid.uuid4())

    # All remaining values become top-level context attributes
    return Context(
        user_id=user_id,
        thread_id=thread_id,
        **configurable,  # Extra fields become top-level attributes
    )


def _process_langchain_messages_stream(
    app: LanggraphChatModel | CompiledStateGraph,
    messages: Sequence[BaseMessage],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> Generator[AIMessageChunk, None, None]:
    """Process LangChain messages in streaming mode using async LangGraph calls internally."""
    import asyncio

    if isinstance(app, LanggraphChatModel):
        app = app.graph

    logger.trace(
        "Processing messages for stream",
        messages_count=len(messages),
        has_custom_inputs=custom_inputs is not None,
    )

    configurable = (custom_inputs or {}).get("configurable", custom_inputs or {})
    context: Context = _configurable_to_context(configurable)

    # Use async astream internally for parallel execution
    async def _async_stream():
        async for nodes, stream_mode, stream_messages in app.astream(
            {"messages": messages},
            context=context,
            config=custom_inputs,
            stream_mode=["messages", "custom"],
            subgraphs=True,
        ):
            nodes: tuple[str, ...]
            stream_mode: str
            stream_messages: Sequence[BaseMessage]
            logger.trace(
                "Stream batch received",
                nodes=nodes,
                stream_mode=stream_mode,
                messages_count=len(stream_messages),
            )
            for message in stream_messages:
                if (
                    isinstance(
                        message,
                        (
                            AIMessageChunk,
                            AIMessage,
                        ),
                    )
                    and message.content
                    and "summarization" not in nodes
                ):
                    yield message

    # Convert async generator to sync generator

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Handle case where no event loop exists (common in some deployment scenarios)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async_gen = _async_stream()

    try:
        while True:
            try:
                item = loop.run_until_complete(async_gen.__anext__())
                yield item
            except StopAsyncIteration:
                break
    finally:
        loop.run_until_complete(async_gen.aclose())


def _process_mlflow_messages(
    app: ChatModel,
    messages: Sequence[ChatMessage],
    custom_inputs: Optional[ChatParams] = None,
) -> ChatCompletionResponse:
    return app.predict(None, messages, custom_inputs)


def _process_mlflow_response_messages(
    app: ResponsesAgent,
    messages: ResponsesAgentRequest,
) -> ResponsesAgentResponse:
    """Process MLflow ResponsesAgent request in batch mode."""
    return app.predict(messages)


def _process_mlflow_messages_stream(
    app: ChatModel,
    messages: Sequence[ChatMessage],
    custom_inputs: Optional[ChatParams] = None,
) -> Generator[ChatCompletionChunk, None, None]:
    for event in app.predict_stream(None, messages, custom_inputs):
        event: ChatCompletionChunk
        yield event


def _process_mlflow_response_messages_stream(
    app: ResponsesAgent,
    messages: ResponsesAgentRequest,
) -> Generator[ResponsesAgentStreamEvent, None, None]:
    """Process MLflow ResponsesAgent request in streaming mode."""
    for event in app.predict_stream(messages):
        event: ResponsesAgentStreamEvent
        yield event


def _process_config_messages(
    app: LanggraphChatModel | LanggraphResponsesAgent,
    messages: dict[str, Any],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> ChatCompletionResponse | ResponsesAgentResponse:
    if isinstance(app, LanggraphChatModel):
        messages: Sequence[ChatMessage] = [ChatMessage(**m) for m in messages]
        params: ChatParams = ChatParams(**{"custom_inputs": custom_inputs})
        return _process_mlflow_messages(app, messages, params)

    elif isinstance(app, LanggraphResponsesAgent):
        input_messages: list[Message] = [Message(**m) for m in messages]
        request = ResponsesAgentRequest(
            input=input_messages, custom_inputs=custom_inputs
        )
        return _process_mlflow_response_messages(app, request)


def _process_config_messages_stream(
    app: LanggraphChatModel | LanggraphResponsesAgent,
    messages: dict[str, Any],
    custom_inputs: dict[str, Any],
) -> Generator[ChatCompletionChunk | ResponsesAgentStreamEvent, None, None]:
    if isinstance(app, LanggraphChatModel):
        messages: Sequence[ChatMessage] = [ChatMessage(**m) for m in messages]
        params: ChatParams = ChatParams(**{"custom_inputs": custom_inputs})

        for event in _process_mlflow_messages_stream(
            app, messages, custom_inputs=params
        ):
            yield event

    elif isinstance(app, LanggraphResponsesAgent):
        input_messages: list[Message] = [Message(**m) for m in messages]
        request = ResponsesAgentRequest(
            input=input_messages, custom_inputs=custom_inputs
        )

        for event in _process_mlflow_response_messages_stream(app, request):
            yield event


def process_messages_stream(
    app: LanggraphChatModel | LanggraphResponsesAgent,
    messages: Sequence[BaseMessage]
    | Sequence[ChatMessage]
    | ResponsesAgentRequest
    | dict[str, Any],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> Generator[
    ChatCompletionChunk | ResponsesAgentStreamEvent | AIMessageChunk, None, None
]:
    """
    Process messages through a ChatAgent in streaming mode.

    Utility function that normalizes message input formats and
    streams the agent's responses as they're generated using async LangGraph calls internally.

    Args:
        app: The ChatAgent to process messages with
        messages: Messages in various formats (list or dict)

    Yields:
        Individual message chunks from the streaming response
    """

    if has_mlflow_responses_messages(messages):
        for event in _process_mlflow_response_messages_stream(app, messages):
            yield event
    elif has_mlflow_messages(messages):
        for event in _process_mlflow_messages_stream(app, messages, custom_inputs):
            yield event
    elif has_langchain_messages(messages):
        for event in _process_langchain_messages_stream(app, messages, custom_inputs):
            yield event
    else:
        for event in _process_config_messages_stream(app, messages, custom_inputs):
            yield event


def process_messages(
    app: LanggraphChatModel | LanggraphResponsesAgent,
    messages: Sequence[BaseMessage]
    | Sequence[ChatMessage]
    | ResponsesAgentRequest
    | dict[str, Any],
    custom_inputs: Optional[dict[str, Any]] = None,
) -> ChatCompletionResponse | ResponsesAgentResponse | dict[str, Any] | Any:
    """
    Process messages through a ChatAgent in batch mode.

    Utility function that normalizes message input formats and
    returns the complete response from the agent using async LangGraph calls internally.

    Args:
        app: The ChatAgent to process messages with
        messages: Messages in various formats (list or dict)

    Returns:
        Complete response from the agent
    """

    if has_mlflow_responses_messages(messages):
        return _process_mlflow_response_messages(app, messages)
    elif has_mlflow_messages(messages):
        return _process_mlflow_messages(app, messages, custom_inputs)
    elif has_langchain_messages(messages):
        return _process_langchain_messages(app, messages, custom_inputs)
    else:
        return _process_config_messages(app, messages, custom_inputs)


def display_graph(app: LanggraphChatModel | CompiledStateGraph) -> None:
    from IPython.display import HTML, Image, display

    if isinstance(app, LanggraphChatModel):
        app = app.graph

    try:
        content = Image(app.get_graph(xray=True).draw_mermaid_png())
    except Exception as e:
        print(e)
        ascii_graph: str = app.get_graph(xray=True).draw_ascii()
        html_content = f"""
    <pre style="font-family: monospace; line-height: 1.2; white-space: pre;">
    {ascii_graph}
    </pre>
    """
        content = HTML(html_content)

    display(content)


def save_image(app: LanggraphChatModel | CompiledStateGraph, path: PathLike) -> None:
    if isinstance(app, LanggraphChatModel):
        app = app.graph

    path = Path(path)
    content = app.get_graph(xray=True).draw_mermaid_png()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
