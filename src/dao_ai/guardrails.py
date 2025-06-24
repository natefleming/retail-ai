from typing import Any, Literal, Optional, Type

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableLike
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps
from loguru import logger
from openevals.llm import create_llm_as_judge

from dao_ai.config import GuardrailModel
from dao_ai.messages import last_ai_message, last_human_message
from dao_ai.state import SharedState


class MessagesWithSteps(MessagesState):
    guardrails_remaining_steps: RemainingSteps


def end_or_reflect(state: MessagesWithSteps) -> Literal[END, "graph"]:
    if state["guardrails_remaining_steps"] < 2:
        return END
    if len(state["messages"]) == 0:
        return END
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        return "graph"
    else:
        return END


def create_reflection_graph(
    graph: CompiledStateGraph,
    reflection: CompiledStateGraph,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
) -> StateGraph:
    logger.debug("Creating reflection graph")
    _state_schema = state_schema or graph.builder.schema

    if "guardrails_remaining_steps" in _state_schema.__annotations__:
        raise ValueError(
            "Has key 'guardrails_remaining_steps' in state_schema, this shadows a built in key"
        )

    if "messages" not in _state_schema.__annotations__:
        raise ValueError("Missing required key 'messages' in state_schema")

    class StateSchema(_state_schema):
        guardrails_remaining_steps: RemainingSteps

    rgraph = StateGraph(StateSchema, config_schema=config_schema)
    rgraph.add_node("graph", graph)
    rgraph.add_node("reflection", reflection)
    rgraph.add_edge(START, "graph")
    rgraph.add_edge("graph", "reflection")
    rgraph.add_conditional_edges("reflection", end_or_reflect)
    return rgraph


def with_guardrails(
    graph: CompiledStateGraph, guardrail: CompiledStateGraph
) -> CompiledStateGraph:
    logger.debug("Creating graph with guardrails")
    return create_reflection_graph(
        graph, guardrail, state_schema=SharedState, config_schema=RunnableConfig
    ).compile()


def judge_node(guardrails: GuardrailModel) -> RunnableLike:
    def judge(state: SharedState, config: RunnableConfig) -> dict[str, BaseMessage]:
        llm: LanguageModelLike = guardrails.model.as_chat_model()

        evaluator = create_llm_as_judge(
            prompt=guardrails.prompt,
            judge=llm,
        )

        ai_message: AIMessage = last_ai_message(state["messages"])
        human_message: HumanMessage = last_human_message(state["messages"])

        logger.debug(f"Evaluating response: {ai_message.content}")
        eval_result = evaluator(
            inputs=human_message.content, outputs=ai_message.content
        )

        if eval_result["score"]:
            logger.debug("✅ Response approved by judge")
            logger.debug(f"Judge's comment: {eval_result['comment']}")
            return
        else:
            # Otherwise, return the judge's critique as a new user message
            logger.warning("⚠️ Judge requested improvements")
            comment: str = eval_result["comment"]
            logger.warning(f"Judge's critique: {comment}")
            content: str = "\n".join([human_message.content, comment])
            return {"messages": [HumanMessage(content=content)]}

    return judge


def reflection_guardrail(guardrails: GuardrailModel) -> CompiledStateGraph:
    judge: CompiledStateGraph = (
        StateGraph(SharedState, config_schema=RunnableConfig)
        .add_node("judge", judge_node(guardrails=guardrails))
        .add_edge(START, "judge")
        .add_edge("judge", END)
        .compile()
    )
    return judge
