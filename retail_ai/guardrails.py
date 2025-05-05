from databricks_langchain import ChatDatabricks
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.state import END, START, CompiledStateGraph, StateGraph
from langgraph_reflection import create_reflection_graph
from loguru import logger
from mlflow.models import ModelConfig
from openevals.llm import create_llm_as_judge

from retail_ai.state import AgentConfig, AgentState
from retail_ai.types import AgentCallable


def with_guardrails(
    graph: CompiledStateGraph, guardrail: CompiledStateGraph
) -> CompiledStateGraph:
    return create_reflection_graph(
        graph, guardrail, state_schema=AgentState, config_schema=AgentConfig
    ).compile()


def judge_node(model_config: ModelConfig) -> AgentCallable:

    model: str = model_config.get("agents").get("recommendation").get("model").get("model_name")

    critique_prompt: str = """
    You are an expert judge evaluating AI responses. Your task is to critique the AI assistant's latest response in the conversation below.

    Evaluate the response based on these criteria:
    1. Accuracy - Is the information correct and factual?
    2. Completeness - Does it fully address the user's query?
    3. Clarity - Is the explanation clear and well-structured?
    4. Helpfulness - Does it provide actionable and useful information?
    5. Safety - Does it avoid harmful or inappropriate content?

    If the response meets ALL criteria satisfactorily, set pass to True.

    If you find ANY issues with the response, do NOT set pass to True. Instead, provide specific and constructive feedback in the comment key and set pass to False.

    Be detailed in your critique so the assistant can understand exactly how to improve.

    <response>
    {outputs}
    </response>
    """

    def judge(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)

        evaluator = create_llm_as_judge(
            prompt=critique_prompt,
            judge=llm,
        )
        eval_result = evaluator(outputs=state["messages"][-1].content, inputs=None)

        if eval_result["score"]:
            logger.debug("✅ Response approved by judge")
            return
        else:
            # Otherwise, return the judge's critique as a new user message
            logger.warning("⚠️ Judge requested improvements")
            return {"messages": [HumanMessage(content=eval_result["comment"])]}

    return judge


def reflection_guardrail(model_config: ModelConfig) -> CompiledStateGraph:
    judge: CompiledStateGraph = (
        StateGraph(AgentState, config_schema=AgentConfig)
        .add_node("judge", judge_node(model_config=model_config))
        .add_edge(START, "judge")
        .add_edge("judge", END)
        .compile()
    )
    return judge
