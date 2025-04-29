from typing import Literal, Optional, Sequence

import mlflow
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain.prompts import PromptTemplate
from langchain_core.documents.base import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableSequence
from langchain_core.vectorstores.base import VectorStore
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from retail_ai.agents import create_genie_agent, create_vector_search_agent
from retail_ai.messages import last_human_message, last_message
from retail_ai.state import AgentConfig, AgentState
from retail_ai.types import AgentCallable

# Define the valid routing destinations for the multi-agent system
allowed_routes: Sequence[str] = (
    "code",              # For programming and technical questions
    "general",           # For general retail questions not fitting other categories
    "genie",             # For database queries about inventory and products
    "vector_search",     # For recommendation and similarity search queries
)

class Router(BaseModel):
    """
    A router that will route the question to the correct agent. 
    
    This Pydantic model defines the structure for the routing decision,
    ensuring that questions are directed to the most appropriate specialized agent:
    
    * Questions about inventory and product information should route to `genie`. 
    * Questions that are most easy solved by code should route to `code`.
    * Questions that about recommendations should route to `vector_search`.
    * All other questions should route to `general`
    """
    route: Literal[tuple(allowed_routes)] = (
        Field(
            default="general", 
            description=f"The route to take. Must be one of {allowed_routes}"
        )
    )


def route_question_node(model_name: str) -> AgentCallable:
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
    def route_question(state: AgentState, config: AgentConfig) -> dict[str, str]:
        # Initialize the language model with low temperature for consistent routing
        llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)
        
        # Create a chain that will output a structured Router object
        chain: RunnableSequence = llm.with_structured_output(Router)
        
        # Extract all messages from the current state
        messages: Sequence[BaseMessage] = state["messages"]
        
        # Get the most recent message from the human user
        last_message: BaseMessage = last_human_message(messages)
        
        # Invoke the chain to determine the appropriate route
        response = chain.invoke([last_message])
        
        # Return the route decision to update the agent state
        return {"route": response.route}

    return route_question
  

def generic_question_node(model_name: str) -> AgentCallable:
    """
    Create a node for handling general retail questions.
    
    This factory function returns a callable that processes general questions
    that don't fit into specialized categories. It uses a simple prompt template
    to generate concise answers to user queries.
    
    Args:
        model_name: Name of the language model to use for answering
        
    Returns:
        An agent callable function that generates a general response
    """
    @mlflow.trace()
    def generic_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        # Initialize the language model with low temperature for factual responses
        llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)
        
        # Create a simple prompt template for general questions
        prompt: PromptTemplate = PromptTemplate.from_template(
            "Give a general and concise answer to the question: {input}"
        )
        
        # Chain the prompt and model together
        chain: RunnableSequence = prompt | llm
        
        # Extract all messages from the current state
        messages: Sequence[BaseMessage] = state["messages"]
        
        # Get the content of the most recent message
        content: str = last_message(messages).content
        
        # Generate a response using the chain
        response = chain.invoke({"input": content})
        
        # Return the response as a message to update the agent state
        return {"messages": [response]}
    
    return generic_question

  
def code_question_node(model_name: str) -> AgentCallable:
    """
    Create a node for handling technical and programming questions.
    
    This factory function returns a callable that processes code-related questions,
    providing detailed step-by-step explanations for technical problems.
    
    Args:
        model_name: Name of the language model to use for technical answers
        
    Returns:
        An agent callable function that generates a detailed technical response
    """
    @mlflow.trace()
    def code_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        # Initialize the language model with low temperature for precise code generation
        llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)
        
        # Create a prompt that instructs the model to act as a software engineer
        prompt: PromptTemplate = PromptTemplate.from_template(
            "You are a software engineer. Answer this question with step by steps details : {input}"
        )
        
        # Chain the prompt and model together
        chain: RunnableSequence = prompt | llm
        
        # Extract all messages from the current state
        messages: Sequence[BaseMessage] = state["messages"]
        
        # Get the content of the most recent message
        content: str = last_message(messages).content
        
        # Generate a technical response using the chain
        response = chain.invoke({"input": content})
        
        # Return the response as a message to update the agent state
        return {"messages": [response]}

    return code_question
  

def vector_search_question_node(
    model_name: str, 
    index_name: str,
    primary_key: str,
    text_column: str,
    doc_uri: str,
    columns: Optional[Sequence[str]] = None,
    search_parameters: dict[str, str] = {},    
) -> AgentCallable:
    """
    Create a node for handling recommendation and search questions using vector search.
    
    This factory function initializes a specialized vector search agent that can find
    similar products, process customer reviews, and generate recommendations based on
    vector similarity in the product database.
    
    Args:
        model_name: Name of the language model to use
        index_name: Name of the vector search index to query
        primary_key: Field name of the document's primary identifier
        text_column: Field name containing the main text content
        doc_uri: Field name for document URI/location
        columns: Specific columns to retrieve (None retrieves all)
        search_parameters: Additional parameters for vector search configuration
        
    Returns:
        An agent callable function that performs vector search and generates responses
    """
    # Initialize the language model for the vector search agent
    model: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)

    # Create a specialized vector search agent with the provided parameters
    vector_search_agent: CompiledStateGraph = create_vector_search_agent(
        model=model,
        index_name=index_name,
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
        columns=columns,
        search_parameters=search_parameters,
    )

    @mlflow.trace()
    def vector_search_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        # Invoke the vector search agent with the current state and configuration
        return vector_search_agent.invoke(input=state, config=config)

    return vector_search_question
  

def genie_question_node(model_name: str, space_id: str) -> AgentCallable:
    """
    Create a node for handling inventory and product database queries.
    
    This factory function initializes a Genie agent that can generate SQL queries,
    fetch inventory information, and answer questions about product availability
    and specifications from the retail database.
    
    Args:
        model_name: Name of the language model to use
        space_id: Databricks workspace ID for Genie integration
        
    Returns:
        An agent callable function that queries databases for product information
    """
    # Initialize the language model for the Genie agent
    model: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)

    # Create a specialized Genie agent for database queries
    genie_agent: CompiledStateGraph = create_genie_agent(
        model=model,
        space_id=space_id
    )

    @mlflow.trace()
    def genie_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        # Invoke the Genie agent with the current state and configuration
        return genie_agent.invoke(input=state, config=config)
    
    return genie_question
  

def summarize_response_node(model_name: str) -> AgentCallable:
    """
    Create a node that summarizes responses for concise customer communication.
    
    This factory function returns a callable that processes the outputs from
    specialized agents and creates a concise, customer-friendly summary as the
    final response.
    
    Args:
        model_name: Name of the language model to use for summarization
        
    Returns:
        An agent callable function that summarizes previous agent outputs
    """
    @mlflow.trace()
    def summarize_response(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        # Initialize the language model for summarization
        llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)
        
        # Create a simple summarization prompt
        prompt: PromptTemplate = PromptTemplate.from_template(
            "Summarize: {input}"
        )
        
        # Chain the prompt and model together
        chain: RunnableSequence = prompt | llm
        
        # Extract all messages from the current state
        messages: Sequence[BaseMessage] = state["messages"]
        
        # Generate a summary of all previous messages
        response = chain.invoke(messages)
        
        # Return the summary as a message to update the agent state
        return {"messages": [response]}

    return summarize_response
  

def retrieve_context_node(
    endpoint: str, 
    index_name: str, 
    search_parameters: dict[str, str]
) -> AgentCallable:
    """
    Create a node that retrieves contextual information from the vector database.
    
    This factory function returns a callable that augments the agent's knowledge
    by fetching relevant documents from a vector store based on the query.
    Used to add product context before routing to specialized agents.
    
    Args:
        endpoint: Databricks Vector Search endpoint URL
        index_name: Name of the vector index to query
        search_parameters: Parameters for the similarity search operation
        
    Returns:
        An agent callable function that retrieves and adds contextual documents
    """
    @mlflow.trace()
    def retrieve_context(state: AgentState, config: AgentConfig) -> dict[str, str]:
        # Extract all messages from the current state
        messages: Sequence[BaseMessage] = state["messages"]
        
        # Get the content of the most recent message
        content: str = last_message(messages).content

        # Initialize the Databricks Vector Search client
        vector_search: VectorStore = DatabricksVectorSearch(
            endpoint=endpoint,
            index_name=index_name,
        )

        # Perform a similarity search to retrieve relevant documents
        context: Sequence[Document] = vector_search.similarity_search(
            query=content, **search_parameters
        )

        # Return the retrieved documents as context
        return {"context": context}

    return retrieve_context
