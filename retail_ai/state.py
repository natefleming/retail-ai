from typing import Sequence, TypedDict

from langchain_core.documents.base import Document
from langgraph.graph import MessagesState


class AgentConfig(TypedDict):
    """
    Configuration parameters for the retail AI agent.

    This TypedDict defines external configuration parameters that can be passed
    to the agent during invocation. It allows for runtime customization of the
    agent's behavior without changing the agent's code.

    Example configurations might include:
    - user_id: Identifier for the current user
    - store_id: Identifier for the relevant retail store location
    - thread_id: Conversation thread identifier for stateful conversations
    - product_categories: Categories to filter for in product searches
    """

    ...  # Fields are defined at runtime based on invocation parameters


class AgentState(MessagesState):
    """
    State representation for the retail AI agent conversation workflow.

    Extends LangGraph's MessagesState to maintain the conversation history while
    adding additional state fields specific to the retail domain. This state is
    passed between nodes in the agent graph and modified during execution.

    Attributes:
        context: Retrieved documents providing relevant product/inventory information
        route: The current routing decision (which specialized agent to use)
        remaining_steps: Counter to limit reasoning steps and prevent infinite loops
    """

    context: Sequence[Document]  # Documents retrieved from vector search
    route: str
    active_agent: str

    is_valid: bool
    validation_error: str
