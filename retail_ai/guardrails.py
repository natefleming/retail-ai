from typing import Any

from databricks_langchain import ChatDatabricks
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import END, START, CompiledStateGraph, StateGraph
from langgraph_reflection import create_reflection_graph
from loguru import logger
from openevals.llm import create_llm_as_judge

from retail_ai.config import GuardrailModel
from retail_ai.messages import last_ai_message, last_human_message
from retail_ai.state import AgentConfig, AgentState
from retail_ai.types import AgentCallable


def with_guardrails(
    graph: CompiledStateGraph, guardrail: CompiledStateGraph
) -> CompiledStateGraph:
    return create_reflection_graph(
        graph, guardrail, state_schema=AgentState, config_schema=AgentConfig
    ).compile()


def judge_node(guardrails: GuardrailModel) -> AgentCallable:
    def judge(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        llm: LanguageModelLike = ChatDatabricks(
            model=guardrails.model.name,
            temperature=guardrails.model.temperature,
        )

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
            return
        else:
            # Otherwise, return the judge's critique as a new user message
            logger.warning("⚠️ Judge requested improvements")
            comment: str = eval_result["comment"]
            logger.warning(f"Judge's critique: {comment}")
            content: str = "\n".join([human_message.content, comment])
            return {"messages": [HumanMessage(content=content)]}

    return judge


def reflection_guardrail(guardrails: dict[str, Any]) -> CompiledStateGraph:
    judge: CompiledStateGraph = (
        StateGraph(AgentState, config_schema=AgentConfig)
        .add_node("judge", judge_node(guardrails=guardrails))
        .add_edge(START, "judge")
        .add_edge("judge", END)
        .compile()
    )
    return judge
