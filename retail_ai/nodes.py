from typing import Any, Literal, Optional, Sequence

import mlflow
from databricks_langchain import ChatDatabricks
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableSequence
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger
from mlflow.models import ModelConfig
from pydantic import BaseModel, Field

from retail_ai.guardrails import reflection_guardrail, with_guardrails
from retail_ai.messages import last_human_message
from retail_ai.state import AgentConfig, AgentState
from retail_ai.tools import search_tool
from retail_ai.types import AgentCallable


def message_validation_node(model_config: ModelConfig) -> AgentCallable:

    @mlflow.trace()
    def message_validation(state: AgentState, config: AgentConfig) -> dict[str, Any]:
        logger.debug(f"state: {state}")

        configurable: dict[str, Any] = config.get("configurable", {})
        validation_errors: list[str] = []

        user_id: Optional[str] = configurable.get("user_id", "")
        if not user_id:
            validation_errors.append("user_id is required")

        store_num: Optional[str] = configurable.get("store_num", "")
        if not store_num:
            validation_errors.append("store_num is required")

        if validation_errors:
            logger.warning(f"Validation errors: {validation_errors}")

        return {
            "user_id": user_id,
            "store_num": store_num,
            "is_valid_config": True
        }

    return message_validation


def router_node(model_config: ModelConfig) -> AgentCallable:
    """
    Create a node that routes questions to the appropriate specialized agent.

    This factory function returns a callable that uses a language model to analyze
    the latest user message and determine which agent should handle it based on content.
    The routing decision is structured through the Router Pydantic model.

    Args:
        model_name: Name of the language model to use for routing decisions

    Returns:
        An agent callable function that updates the state with the routing decision
    """

    model: str = model_config.get("agents").get("router").get("model").get("model_name")
    prompt: str = model_config.get("agents").get("router").get("prompt")
    allowed_routes: Sequence[str] = (
        model_config.get("agents").get("router").get("allowed_routes")
    )
    default_route: str = model_config.get("agents").get("router").get("default_route")

    @mlflow.trace()
    def router(state: AgentState, config: AgentConfig) -> dict[str, str]:

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)

        class Router(BaseModel):

            route: Literal[tuple(allowed_routes)] = Field(
                default=default_route,
                description=f"The route to take. Must be one of {allowed_routes}",
            )

        Router.__doc__ = prompt

        chain: RunnableSequence = llm.with_structured_output(Router)

        # Extract all messages from the current state
        messages: Sequence[BaseMessage] = state["messages"]

        # Get the most recent message from the human user
        last_message: BaseMessage = last_human_message(messages)

        # Invoke the chain to determine the appropriate route
        response = chain.invoke([last_message])

        # Return the route decision to update the agent state
        return {"route": response.route}

    return router


def general_node(model_config: ModelConfig) -> AgentCallable:

    model: str = model_config.get("agents").get("general").get("model").get("model_name")
    prompt: str = model_config.get("agents").get("general").get("prompt")
    guardrails: Sequence[dict[str, Any]] = model_config.get("agents").get("general").get("guardrails") or []

    @mlflow.trace()
    def general(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)

        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = {
            "user_id": state["user_id"],
            "store_num": state["store_num"],
        }
        system_prompt: str = prompt_template.format(**configurable)

        tools = []

        agent: CompiledStateGraph = create_react_agent(
            model=llm, 
            prompt=system_prompt, 
            tools=tools, 
        )

        for guardrail_definition in guardrails:
            guardrail: CompiledStateGraph = reflection_guardrail(guardrail_definition)
            agent = with_guardrails(agent, guardrail)

        return agent

    return general


def product_node(model_config: ModelConfig) -> AgentCallable:

    model: str = model_config.get("agents").get("product").get("model").get("model_name")
    prompt: str = model_config.get("agents").get("product").get("prompt")
    guardrails: dict[str, Any] = model_config.get("agents").get("product").get("guardrails") or []

    @mlflow.trace()
    def product(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)

        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = {
            "user_id": state["user_id"],
            "store_num": state["store_num"],
        }
        system_prompt: str = prompt_template.format(**configurable)

        tools = []

        agent: CompiledStateGraph = create_react_agent(
            model=llm, 
            prompt=system_prompt, 
            tools=tools, 
        )

        for guardrail_definition in guardrails:
            guardrail: CompiledStateGraph = reflection_guardrail(guardrail_definition)
            agent = with_guardrails(agent, guardrail)

        return agent

    return product


def inventory_node(model_config: ModelConfig) -> AgentCallable:

    model: str = model_config.get("agents").get("inventory").get("model").get("model_name")
    prompt: str = model_config.get("agents").get("inventory").get("prompt")
    guardrails: dict[str, Any] = model_config.get("agents").get("inventory").get("guardrails") or []

    @mlflow.trace()
    def inventory(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)

        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = {
            "user_id": state["user_id"],
            "store_num": state["store_num"],
        }
        system_prompt: str = prompt_template.format(**configurable)

        tools = []

        agent: CompiledStateGraph = create_react_agent(
            model=llm, 
            prompt=system_prompt, 
            tools=tools, 
        )

        for guardrail_definition in guardrails:
            guardrail: CompiledStateGraph = reflection_guardrail(guardrail_definition)
            agent = with_guardrails(agent, guardrail)

        return agent

    return inventory


def comparison_node(model_config: ModelConfig) -> AgentCallable:
    model: str = model_config.get("agents").get("comparison").get("model").get("model_name")
    prompt: str = model_config.get("agents").get("comparison").get("prompt")
    guardrails: dict[str, Any] = model_config.get("agents").get("comparison").get("guardrails") or []

    @mlflow.trace()
    def comparison(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)

        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = {
            "user_id": state["user_id"],
            "store_num": state["store_num"],
        }
        system_prompt: str = prompt_template.format(**configurable)

        tools = [] 

        agent: CompiledStateGraph = create_react_agent(
            model=llm, 
            prompt=system_prompt, 
            tools=tools, 
        )

        for guardrail_definition in guardrails:
            guardrail: CompiledStateGraph = reflection_guardrail(guardrail_definition)
            agent = with_guardrails(agent, guardrail)

        return agent

    return comparison


def orders_node(model_config: ModelConfig) -> AgentCallable:

    model: str = model_config.get("agents").get("orders").get("model").get("model_name")
    prompt: str = model_config.get("agents").get("orders").get("prompt")
    guardrails: dict[str, Any] = model_config.get("agents").get("orders").get("guardrails")

    @mlflow.trace()
    def orders(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)

        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = {
            "user_id": state["user_id"],
            "store_num": state["store_num"],
        }
        system_prompt: str = prompt_template.format(**configurable)

        tools = [] 
        
        agent: CompiledStateGraph = create_react_agent(
            model=llm, 
            prompt=system_prompt, 
            tools=tools, 
        )

        for guardrail_definition in guardrails:
            guardrail: CompiledStateGraph = reflection_guardrail(guardrail_definition)
            agent = with_guardrails(agent, guardrail)

        return agent

    return orders


def diy_node(model_config: ModelConfig) -> AgentCallable:

    model: str = model_config.get("agents").get("diy").get("model").get("model_name")
    prompt: str = model_config.get("agents").get("diy").get("prompt")
    guardrails: dict[str, Any] = model_config.get("agents").get("diy").get("guardrails") or []

    @mlflow.trace()
    def diy(state: AgentState, config: AgentConfig) -> CompiledStateGraph:

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)

        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = {
            "user_id": state["user_id"],
            "store_num": state["store_num"],
        }
        system_prompt: str = prompt_template.format(**configurable)

        tools = [search_tool(model_config)]

        agent: CompiledStateGraph = create_react_agent(
            model=llm, 
            prompt=system_prompt, 
            tools=tools, 
        )

        for guardrail_definition in guardrails:
            guardrail: CompiledStateGraph = reflection_guardrail(guardrail_definition)
            agent = with_guardrails(agent, guardrail)

        return agent
    
    return diy


def recommendation_node(model_config: ModelConfig) -> AgentCallable:

    model: str = model_config.get("agents").get("recommendation").get("model").get("model_name")
    prompt: str = model_config.get("agents").get("recommendation").get("prompt")
    guardrails: dict[str, Any] = model_config.get("agents").get("recommendation").get("guardrails") or []

    @mlflow.trace()
    def recommendation(
        state: AgentState, config: AgentConfig
    ) -> dict[str, BaseMessage]:

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)

        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = {
            "user_id": state["user_id"],
            "store_num": state["store_num"],
        }
        system_prompt: str = prompt_template.format(**configurable)

        tools = [] 

        agent: CompiledStateGraph = create_react_agent(
            model=llm, 
            prompt=system_prompt, 
            tools=tools, 
        )

        for guardrail_definition in guardrails:
            guardrail: CompiledStateGraph = reflection_guardrail(guardrail_definition)
            agent = with_guardrails(agent, guardrail)

        return agent

    return recommendation


def process_images_node(model_config: ModelConfig) -> AgentCallable:

    model: str = model_config.get("agents").get("process_image").get("model").get("model_name")
    prompt: str = model_config.get("agents").get("process_image").get("prompt")

    @mlflow.trace()
    def process_images(
        state: AgentState, config: AgentConfig
    ) -> dict[str, BaseMessage]:
        logger.debug("process_images")

        class ImageDetails(BaseModel):
            summary: str = Field(..., description="The summary of the image")
            product_names: Optional[Sequence[str]] = Field(..., description="The name of the product", default_factory=list)
            upcs: Optional[Sequence[str]] = Field(..., description="The UPC of the image", default_factory=list)

        class ImageProcessor(BaseModel):
            prompts: Sequence[str] = Field(..., description="The prompts to use to process the image", default_factory=list)
            image_details: Sequence[ImageDetails] = Field(..., description="The details of the image", default_factory=list)
 
        ImageProcessor.__doc__ = prompt


        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)

        last_message: HumanMessage = last_human_message(state["messages"])
        messages: Sequence[BaseMessage] = [last_message]

        llm_with_schema: LanguageModelLike = llm.with_structured_output(ImageProcessor)

        image_processor: ImageProcessor = llm_with_schema.invoke(input=messages)

        logger.debug(f"image_processor: {image_processor}")

        response_messages: Sequence[BaseMessage] = [
            RemoveMessage(last_message.id),
            HumanMessage(content=image_processor.model_dump_json()),
        ]

        return {"messages": response_messages}

    return process_images
