from typing import Any, Literal, Sequence

import mlflow
from databricks_langchain import ChatDatabricks
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.runnables import RunnableSequence
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger
from mlflow.models import ModelConfig
from openevals.llm import create_llm_as_judge
from pydantic import BaseModel, Field

from retail_ai.messages import (last_human_message)
from retail_ai.state import AgentConfig, AgentState
from retail_ai.types import AgentCallable


def message_validation_node(model_config: ModelConfig) -> AgentCallable:

    @mlflow.trace()
    def message_validation(state: AgentState, config: AgentConfig) -> dict[str, Any]:
        logger.debug(f"state: {state}")
     
        configurable: dict[str, Any] = config.get("configurable", {})
        validation_errors: list[str] = []
        
        # Check user_id
        if "user_id" not in configurable:
            validation_errors.append("Missing required configuration: user_id (Customer identifier)")
        elif not configurable["user_id"]:
            validation_errors.append("User ID cannot be empty")
        elif not isinstance(configurable["user_id"], (str, int)):
            validation_errors.append("User ID must be a string or integer")
            
        # Check store_num
        if "store_num" not in configurable:
            validation_errors.append("Missing required configuration: store_num (Store location identifier)")
        elif configurable["store_num"] is None or configurable["store_num"] == "":
            validation_errors.append("Store number cannot be empty")
        elif not isinstance(configurable["store_num"], (str, int)):
            validation_errors.append("Store number must be a string or integer")
            
        # Check scd_ids
        if "scd_ids" not in configurable:
            validation_errors.append("Missing required configuration: scd_ids (Product identifiers)")
        elif not configurable["scd_ids"]:
            validation_errors.append("Product identifiers list cannot be empty")
        elif not isinstance(configurable["scd_ids"], (list, tuple, set)):
            validation_errors.append("Product identifiers must be a list, tuple, or set")
        
        # If validation errors exist, return helpful message
        if validation_errors:
            content: str = "## Configuration Validation Failed\n\n" + "\n".join([f"- {error}" for error in validation_errors])
            content += "\n\nPlease provide all required configuration values to proceed with your request."
            content += "\n\n### Example of a valid configuration payload:\n"
            content += "```json\n{\n    \"configurable\": {\n        \"user_id\": \"customer_12345\",\n        \"store_num\": 789,\n        \"scd_ids\": [\"SKU123\", \"SKU456\", \"SKU789\"]\n    }\n}\n```"
            content += "\n\nIf you have any questions, please contact support."
            message: AIMessage = AIMessage(content=content)
            logger.error(f"Validation failed: {content}")
            return {
                "messages": [message],
                "is_valid_config": False
            }

        # All validations passed
        return {
            "is_valid_config": True
        }

    return message_validation


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
    @mlflow.trace()
    def router(state: AgentState, config: AgentConfig) -> dict[str, str]:
   
        model: str = model_config.get("agents").get("router").get("model_name")
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        
        prompt: str = model_config.get("agents").get("router").get("prompt")
        allowed_routes: Sequence[str] = model_config.get("agents").get("router").get("allowed_routes")
        default_route = model_config.get("agents").get("router").get("default_route")

        class Router(BaseModel):

            route: Literal[tuple(allowed_routes)] = (
                Field(
                    default=default_route, 
                    description=f"The route to take. Must be one of {allowed_routes}"
                )
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

    @mlflow.trace()
    def general(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        model: str = model_config.get("agents").get("general").get("model_name")
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        
        prompt: str = model_config.get("agents").get("general").get("prompt")
        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = config.get("configurable", {})
        system_prompt: str = prompt_template.format(
            **configurable
        )

        system_message: SystemMessage = SystemMessage(content=system_prompt)
        
        messages: Sequence[BaseMessage] = state["messages"]
        
        messages = [system_message] + messages

        agent: CompiledStateGraph = create_react_agent(model=llm, prompt=system_prompt, tools=[])
        return agent
    
    return general


def product_node(model_config: ModelConfig) -> AgentCallable:

    @mlflow.trace()
    def product(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        model: str = model_config.get("agents").get("product").get("model_name")
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        
        prompt: str = model_config.get("agents").get("product").get("prompt")
        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = config.get("configurable", {})
        system_prompt: str = prompt_template.format(
            **configurable
        )

        system_message: SystemMessage = SystemMessage(content=system_prompt)
        
        messages: Sequence[BaseMessage] = state["messages"]
        
        messages = [system_message] + messages

        agent: CompiledStateGraph = create_react_agent(model=llm, prompt=system_prompt, tools=[])
        return agent
    
    return product

def inventory_node(model_config: ModelConfig) -> AgentCallable:

    @mlflow.trace()
    def inventory(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        model: str = model_config.get("agents").get("inventory").get("model_name")
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        
        prompt: str = model_config.get("agents").get("inventory").get("prompt")
        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = config.get("configurable", {})
        system_prompt: str = prompt_template.format(
            **configurable
        )

        system_message: SystemMessage = SystemMessage(content=system_prompt)
        
        messages: Sequence[BaseMessage] = state["messages"]
        
        messages = [system_message] + messages

        agent: CompiledStateGraph = create_react_agent(model=llm, prompt=system_prompt, tools=[])
        return agent
    
    return inventory

def comparison_node(model_config: ModelConfig) -> AgentCallable:

    @mlflow.trace()
    def comparison(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        model: str = model_config.get("agents").get("comparison").get("model_name")
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        
        prompt: str = model_config.get("agents").get("comparison").get("prompt")
        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = config.get("configurable", {})
        system_prompt: str = prompt_template.format(
            **configurable
        )

        system_message: SystemMessage = SystemMessage(content=system_prompt)
        
        messages: Sequence[BaseMessage] = state["messages"]
        
        messages = [system_message] + messages

        agent: CompiledStateGraph = create_react_agent(model=llm, prompt=system_prompt, tools=[])
        return agent
        
    return comparison

def orders_node(model_config: ModelConfig) -> AgentCallable:

    @mlflow.trace()
    def orders(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        model: str = model_config.get("agents").get("orders").get("model_name")
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        
        prompt: str = model_config.get("agents").get("orders").get("prompt")
        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = config.get("configurable", {})
        system_prompt: str = prompt_template.format(
            **configurable
        )

        system_message: SystemMessage = SystemMessage(content=system_prompt)
        
        messages: Sequence[BaseMessage] = state["messages"]
        
        messages = [system_message] + messages

        agent: CompiledStateGraph = create_react_agent(model=llm, prompt=system_prompt, tools=[])
        return agent
    
    return orders

def diy_node(model_config: ModelConfig) -> AgentCallable:

    @mlflow.trace()
    def diy(state: AgentState, config: AgentConfig) -> CompiledStateGraph:
        model: str = model_config.get("agents").get("diy").get("model_name")
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        
        prompt: str = model_config.get("agents").get("diy").get("prompt")
        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = config.get("configurable", {})
        system_prompt: str = prompt_template.format(
            **configurable
        )

        system_message: SystemMessage = SystemMessage(content=system_prompt)
        
        messages: Sequence[BaseMessage] = state["messages"]
        
        messages = [system_message] + messages

        agent: CompiledStateGraph = create_react_agent(model=llm, prompt=system_prompt, tools=[])
        return agent
    
    return diy

def recommendation_node(model_config: ModelConfig) -> AgentCallable:

    @mlflow.trace()
    def recommendation(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        model: str = model_config.get("agents").get("recommendation").get("model_name")
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        
        prompt: str = model_config.get("agents").get("recommendation").get("prompt")
        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = config.get("configurable", {})
        system_prompt: str = prompt_template.format(
            **configurable
        )

        system_message: SystemMessage = SystemMessage(content=system_prompt)
        
        messages: Sequence[BaseMessage] = state["messages"]
        
        messages = [system_message] + messages

        agent: CompiledStateGraph = create_react_agent(model=llm, prompt=system_prompt, tools=[])
        return agent
    
    return recommendation


def judge_node(model_config: ModelConfig) -> AgentCallable:

    critique_prompt = """
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

        model: str = model_config.get("agents").get("recommendation").get("model_name")
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