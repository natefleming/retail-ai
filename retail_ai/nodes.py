from typing import Any, Literal, Sequence

import mlflow
from databricks_langchain import ChatDatabricks
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableSequence
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger
from mlflow.models import ModelConfig
from pydantic import BaseModel, Field

from retail_ai.guardrails import reflection_guardrail, with_guardrails
from retail_ai.messages import last_human_message
from retail_ai.state import AgentConfig, AgentState
from retail_ai.types import AgentCallable
from retail_ai.tools import search_tool

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

    model: str = model_config.get("agents").get("router").get("model_name")
    prompt: str = model_config.get("agents").get("router").get("prompt")
    allowed_routes: Sequence[str] = model_config.get("agents").get("router").get("allowed_routes")
    default_route = model_config.get("agents").get("router").get("default_route")

    @mlflow.trace()
    def router(state: AgentState, config: AgentConfig) -> dict[str, str]:
   
        
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        
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

    model: str = model_config.get("agents").get("general").get("model_name")
    prompt: str = model_config.get("agents").get("general").get("prompt")

    @mlflow.trace()
    def general(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        

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

    model: str = model_config.get("agents").get("product").get("model_name")
    prompt: str = model_config.get("agents").get("product").get("prompt")

    @mlflow.trace()
    def product(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
       
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        

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

    model: str = model_config.get("agents").get("inventory").get("model_name")
    prompt: str = model_config.get("agents").get("inventory").get("prompt")

    @mlflow.trace()
    def inventory(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
    
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        

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
    model: str = model_config.get("agents").get("comparison").get("model_name")
    prompt: str = model_config.get("agents").get("comparison").get("prompt")

    @mlflow.trace()
    def comparison(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        

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

    model: str = model_config.get("agents").get("orders").get("model_name")
    prompt: str = model_config.get("agents").get("orders").get("prompt")

    @mlflow.trace()
    def orders(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        

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

    model: str = model_config.get("agents").get("diy").get("model_name")
    prompt: str = model_config.get("agents").get("diy").get("prompt")

    @mlflow.trace()
    def diy(state: AgentState, config: AgentConfig) -> CompiledStateGraph:

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        

        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        configurable: dict[str, Any] = config.get("configurable", {})
        system_prompt: str = prompt_template.format(
            **configurable
        )

        system_message: SystemMessage = SystemMessage(content=system_prompt)
        
        messages: Sequence[BaseMessage] = state["messages"]
        
        messages = [system_message] + messages

        tools = [
            search_tool(model_config)
        ]

        agent: CompiledStateGraph = create_react_agent(model=llm, prompt=system_prompt, tools=tools)


        guardrail: CompiledStateGraph = reflection_guardrail(model_config=model_config)
        agent_with_guard_rail: CompiledStateGraph = with_guardrails(agent, guardrail)

        return agent_with_guard_rail
    
    return diy

def recommendation_node(model_config: ModelConfig) -> AgentCallable:

    model: str = model_config.get("agents").get("recommendation").get("model_name")
    prompt: str = model_config.get("agents").get("recommendation").get("prompt")

    @mlflow.trace()
    def recommendation(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        
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

def process_images_node(model_config: ModelConfig) -> AgentCallable:

    model: str = model_config.get("agents").get("process_image").get("model_name")
    prompt: str = model_config.get("agents").get("process_image").get("prompt")

    @mlflow.trace()
    def  process_images(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        logger.debug("process_images")

        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        

        prompt_template: PromptTemplate = PromptTemplate.from_template(prompt)
        system_prompt: str = prompt_template.format()

        system_message: SystemMessage = SystemMessage(content=system_prompt)
        message: HumanMessage = last_human_message(state["messages"])
        messages: Sequence[BaseMessage] = [system_message, message]

        response: AIMessage = llm.invoke(input=messages)

        image_summary: str = response.content
        
        logger.debug("image_summary: {image_summary}")

        return {"image_summary": image_summary}
    
    return  process_images



