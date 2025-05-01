from typing import Any, Literal, Optional, Sequence

import mlflow
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain.prompts import PromptTemplate

from langchain_core.documents.base import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableSequence
from langchain_core.vectorstores.base import VectorStore
from langgraph.graph.state import CompiledStateGraph, StateGraph, END
from loguru import logger
from mlflow.models import ModelConfig
from pydantic import BaseModel, Field

from retail_ai.agents import (create_arma_agent, create_genie_agent,
                              create_vector_search_agent)
from retail_ai.messages import last_human_message, last_message, last_ai_message
from retail_ai.state import AgentConfig, AgentState
from retail_ai.types import AgentCallable
from retail_ai.tools import (
    find_allowable_classifications,
)


def message_validation_node(model_config: ModelConfig) -> AgentCallable:

    @mlflow.trace()
    def message_validation(state: AgentState, config: AgentConfig) -> dict[str, Any]:
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


        logger.debug(f"Sending messages to model: {messages}")
        response = llm.invoke(messages)
        logger.debug(f"Received response from model: {response}")

        # Return the response as a message to update the agent state
        return {"messages": [response]}
    
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


        logger.debug(f"Sending messages to model: {messages}")
        response = llm.invoke(messages)
        logger.debug(f"Received response from model: {response}")
        
        # Return the response as a message to update the agent state
        return {"messages": [response]}
    
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


        logger.debug(f"Sending messages to model: {messages}")
        response = llm.invoke(messages)
        logger.debug(f"Received response from model: {response}")
        
        # Return the response as a message to update the agent state
        return {"messages": [response]}
    
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


        logger.debug(f"Sending messages to model: {messages}")
        response = llm.invoke(messages)
        logger.debug(f"Received response from model: {response}")
        
        # Return the response as a message to update the agent state
        return {"messages": [response]}
        
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


        logger.debug(f"Sending messages to model: {messages}")
        response = llm.invoke(messages)
        logger.debug(f"Received response from model: {response}")
        
        # Return the response as a message to update the agent state
        return {"messages": [response]}
    
    return orders

def diy_node(model_config: ModelConfig) -> AgentCallable:

    @mlflow.trace()
    def diy(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
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


        logger.debug(f"Sending messages to model: {messages}")
        response = llm.invoke(messages)
        logger.debug(f"Received response from model: {response}")
        
        # Return the response as a message to update the agent state
        return {"messages": [response]}
    
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


        logger.debug(f"Sending messages to model: {messages}")
        response = llm.invoke(messages)
        logger.debug(f"Received response from model: {response}")
        
        # Return the response as a message to update the agent state
        return {"messages": [response]}
    
    return recommendation



def factuality_judge_node(model_config: ModelConfig) -> AgentCallable:
    """
    Create a node that evaluates and refines responses for factual accuracy.
    
    This follows the DSPy.Refine pattern where responses are evaluated, and if
    they don't meet quality standards, they're iteratively refined until they do
    or until we reach a maximum number of attempts.
    
    Args:
        model_config: Configuration for models and prompts
        
    Returns:
        A compiled LangGraph workflow for factuality checking and refinement
    """
    # Define the evaluation schema for factuality checking
    class FactualityJudge(BaseModel):
        """Structured evaluation of factual accuracy."""
        is_factual: bool = Field(description="Whether the statement is factually correct")
        reason: str = Field(description="Explanation of factual correctness or issues identified")

    # Helper functions to extract configuration values
    def max_retries_from(config: AgentConfig) -> int:
        DEFAULT_MAX_RETRIES: int = 3
        return config.get("configurable", {}).get("max_retries", DEFAULT_MAX_RETRIES)
    
    def retry_count_from(state: AgentState) -> int:
        return state.get("retry_count", 0)
    
    @mlflow.trace
    def factuality_judge(state: AgentState, config: AgentConfig) -> dict[str, Any]:
        """
        Evaluate the factuality of the latest AI response.
        
        Args:
            state: Current state containing message history
            config: Configuration parameters
            
        Returns:
            dictionary with factuality assessment and message
        """
        # Get model configuration for the judge
        model: str = model_config.to_dict().get("agents", {}).get("factuality", {}).get("model_name")
        if not model:
            model = model_config.to_dict().get("llms", {}).get("model_name")
        
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        
        # Get the latest AI message to evaluate
        last_message: AIMessage = last_ai_message(state["messages"])
        if not last_message:
            logger.warning("No AI message found to evaluate")
            return {
                "is_factual_response": True,  # Default to true if no message to check
                "factuality_message": None
            }
        
        # Create a system prompt for evaluation
        system_prompt = model_config.to_dict().get("agents", {}).get("factuality", {}).get("evaluation_prompt", 
            """You are an expert factuality judge. Carefully analyze the following statement 
            and determine if it contains any factual errors or inaccuracies. Focus only on 
            verifiable facts, not opinions or subjective statements.""")
        
        system_message = SystemMessage(content=system_prompt)
        
        # Create the evaluation message
        eval_message = HumanMessage(content=f"Evaluate this statement for factual accuracy: {last_message.content}")
        
        # Configure the LLM for structured output
        llm_with_schema = llm.with_structured_output(FactualityJudge)
        
        # Evaluate the statement
        evaluation = llm_with_schema.invoke([system_message, eval_message])
        
        logger.debug(f"Factuality evaluation: {evaluation.is_factual}, Reason: {evaluation.reason}")
        
        return {
            "is_factual_response": evaluation.is_factual,
            "factuality_message": evaluation.reason
        }

    @mlflow.trace
    def fix_statement(state: AgentState, config: AgentConfig) -> dict[str, Any]:
        """
        Refine a response to fix factual inaccuracies.
        
        Args:
            state: Current state containing message history and factuality feedback
            config: Configuration parameters
            
        Returns:
            dictionary with updated message and retry count
        """
        # Get model configuration for refinement
        model: str = model_config.to_dict().get("agents", {}).get("factuality", {}).get("model_name")
        if not model:
            model = model_config.to_dict().get("llms", {}).get("model_name")
            
        llm: LanguageModelLike = ChatDatabricks(model=model, temperature=0.1)
        
        logger.info(f"Refining response (attempt {state.get('retry_count', 0) + 1})")
        
        # Get the original response that needs fixing
        last_message: AIMessage = last_ai_message(state["messages"])
        
        # Create a system prompt for refinement
        system_prompt = model_config.to_dict().get("agents", {}).get("factuality", {}).get("refinement_prompt", 
            """You are an expert at providing accurate information. Revise the previous response
            to ensure it's factually correct, addressing the specific issues identified.""")
        
        system_message = SystemMessage(content=system_prompt)
        
        # Create a message with refinement instructions
        fix_message = HumanMessage(
            content=f"""
            The following response contains factual errors:
            
            {last_message.content}
            
            Issue identified: {state["factuality_message"]}
            
            Please provide a revised response that corrects these factual errors while maintaining 
            the helpful nature of the original response.
            """
        )
        
        # Generate improved response
        response = llm.invoke([system_message, fix_message])
        
        # Increment retry counter
        retry_count = retry_count_from(state) + 1
        
        return {
            "messages": [response],  # Add refined response to messages
            "retry_count": retry_count
        }

    @mlflow.trace
    def judge_router(state: AgentState, config: AgentConfig) -> str:
        """
        Determine next step based on factuality results and retry count.
        
        Args:
            state: Current state with factuality assessment
            config: Configuration parameters including max retries
            
        Returns:
            The next node to execute or END if done
        """
        max_retries: int = max_retries_from(config)
        
        if state["is_factual_response"]:
            logger.info("Response is factually correct - finishing")
            return END
        elif state.get("retry_count", 0) >= max_retries:
            logger.warning(f"Reached max retries ({max_retries}) - handling final state")
            return "max_retries_reached"
        else:
            logger.info(f"Response needs refinement (attempt {state.get('retry_count', 0) + 1}/{max_retries})")
            return "fix_statement"
    
    @mlflow.trace
    def handle_max_retries(state: AgentState, config: AgentConfig) -> dict[str, str]:
        """
        Handle the case where maximum retries are reached without success.
        
        Args:
            state: Current state with retry count and error messages
            config: Configuration parameters
            
        Returns:
            dictionary with final status message
        """
        logger.warning(f"Maximum retry attempts ({state.get('retry_count', 0)}) reached without resolution")
        
        # Add a disclaimer message to the existing response
        disclaimer = f"""
        Note: We've made our best effort to provide accurate information, but some details may require verification.
        Specifically: {state["factuality_message"]}
        """
        
        return {
            "factuality_message": disclaimer
        }

    # Create the workflow graph
    workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)

    # Add nodes
    workflow.add_node("factuality_judge", factuality_judge)
    workflow.add_node("fix_statement", fix_statement)
    workflow.add_node("handle_max_retries", handle_max_retries)
    
    # Add conditional routing based on factuality and retry count
    workflow.add_conditional_edges(
        "factuality_judge",
        judge_router,
        {
            "fix_statement": "fix_statement",
            END: END,  # Terminate when factual
            "max_retries_reached": "handle_max_retries"
        }
    ) 
    
    # Add remaining edges
    workflow.add_edge("fix_statement", "factuality_judge")  # Re-evaluate after fixing
    workflow.add_edge("handle_max_retries", END)  # End after handling max retries
    
    # Set entry and exit points
    workflow.set_entry_point("factuality_judge")
    
    # Compile and return the workflow
    return workflow.compile()