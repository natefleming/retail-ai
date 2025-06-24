from langgraph.graph import MessagesState
from langgraph.managed import RemainingSteps


class IncomingState(MessagesState): ...


class OutgoingState(MessagesState):
    is_valid: bool
    message_error: str


class SharedState(MessagesState):
    """
    State representation for the DAO AI agent conversation workflow.

    Extends LangGraph's MessagesState to maintain the conversation history while
    adding additional state fields specific to the DAO domain. This state is
    passed between nodes in the agent graph and modified during execution.
    """

    context: str  # short term/long term memory

    active_agent: str  # langgraph-swarm
    remaining_steps: RemainingSteps  # langgraph-supervisor

    summary: str  # summarization node

    is_valid: bool  # message validation node
    message_error: str
