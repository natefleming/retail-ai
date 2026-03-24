import pytest
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from dao_ai.config import AppConfig
from dao_ai.state import (
    AgentState,
    GenieSpaceState,
    GenieState,
    SessionState,
    merge_session,
)


@pytest.mark.unit
def test_agent_config_creation() -> None:
    """Test creating an RunnableConfig instance."""
    config = RunnableConfig()
    assert isinstance(config, dict)


@pytest.mark.unit
def test_agent_config_with_fields() -> None:
    """Test RunnableConfig with custom fields."""
    config = RunnableConfig(
        user_id="user123", store_num="store456", is_valid_config=True
    )

    assert config["user_id"] == "user123"
    assert config["store_num"] == "store456"
    assert config["is_valid_config"] is True


@pytest.mark.unit
def test_agent_state_creation() -> None:
    """Test creating an AgentState instance."""
    test_document = Document(page_content="Test content", metadata={"source": "test"})

    state = AgentState(
        messages=[HumanMessage(content="Hello")],
        context=[test_document],
        route="search",
        active_agent="product_agent",
        is_valid_config=True,
        user_id="user123",
        store_num="store456",
    )

    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)
    assert state["messages"][0].content == "Hello"
    assert len(state["context"]) == 1
    assert state["context"][0].page_content == "Test content"
    assert state["route"] == "search"
    assert state["active_agent"] == "product_agent"
    assert state["user_id"] == "user123"
    assert state["store_num"] == "store456"
    assert state["is_valid_config"] is True


@pytest.mark.unit
def test_agent_state_inherits_messages_state() -> None:
    """Test that AgentState properly inherits from MessagesState."""
    state = AgentState(
        messages=[
            HumanMessage(content="First message"),
            HumanMessage(content="Second message"),
        ],
        context=[],
        route="default",
        active_agent="main",
        is_valid_config=False,
        user_id="",
        store_num="",
    )

    # Should behave like a MessagesState
    assert "messages" in state
    assert len(state["messages"]) == 2
    assert state["messages"][0].content == "First message"
    assert state["messages"][1].content == "Second message"


@pytest.mark.unit
def test_agent_state_with_empty_context() -> None:
    """Test AgentState with empty context list."""
    state = AgentState(
        messages=[],
        context=[],
        route="",
        active_agent="",
        is_valid_config=False,
        user_id="",
        store_num="",
    )

    assert len(state["context"]) == 0
    assert isinstance(state["context"], list)


@pytest.mark.unit
def test_agent_state_with_multiple_documents() -> None:
    """Test AgentState with multiple documents in context."""
    docs = [
        Document(page_content="Doc 1", metadata={"id": 1}),
        Document(page_content="Doc 2", metadata={"id": 2}),
        Document(page_content="Doc 3", metadata={"id": 3}),
    ]

    state = AgentState(
        messages=[],
        context=docs,
        route="vector_search",
        active_agent="search_agent",
        is_valid_config=True,
        user_id="user789",
        store_num="store123",
    )

    assert len(state["context"]) == 3
    assert all(isinstance(doc, Document) for doc in state["context"])
    assert state["context"][0].metadata["id"] == 1
    assert state["context"][1].metadata["id"] == 2
    assert state["context"][2].metadata["id"] == 3


@pytest.mark.unit
def test_agent_config_integration_with_app_config(config: AppConfig) -> None:
    """Test that RunnableConfig works with the existing config fixture."""
    agent_config = RunnableConfig(
        user_id="test_user", store_num="store001", is_valid_config=config is not None
    )

    # Should work with the existing config
    assert agent_config["is_valid_config"] is True
    assert agent_config["user_id"] == "test_user"
    assert agent_config["store_num"] == "store001"


# ---------------------------------------------------------------------------
# merge_session reducer tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_session_disjoint_spaces() -> None:
    """merge_session should combine spaces from both SessionState values."""
    space_a = GenieSpaceState(conversation_id="conv-a")
    space_b = GenieSpaceState(conversation_id="conv-b")

    current = SessionState(genie=GenieState(spaces={"space_1": space_a}))
    new = SessionState(genie=GenieState(spaces={"space_2": space_b}))

    merged = merge_session(current, new)

    assert "space_1" in merged.genie.spaces
    assert "space_2" in merged.genie.spaces
    assert merged.genie.spaces["space_1"].conversation_id == "conv-a"
    assert merged.genie.spaces["space_2"].conversation_id == "conv-b"


@pytest.mark.unit
def test_merge_session_overlapping_spaces_takes_new() -> None:
    """When both sides contain the same space_id, the new value wins."""
    old_space = GenieSpaceState(conversation_id="old-conv", last_query="old query")
    new_space = GenieSpaceState(conversation_id="new-conv", last_query="new query")

    current = SessionState(genie=GenieState(spaces={"space_1": old_space}))
    new = SessionState(genie=GenieState(spaces={"space_1": new_space}))

    merged = merge_session(current, new)

    assert merged.genie.spaces["space_1"].conversation_id == "new-conv"
    assert merged.genie.spaces["space_1"].last_query == "new query"


@pytest.mark.unit
def test_merge_session_empty_current() -> None:
    """Merging into an empty current should preserve the new state."""
    space = GenieSpaceState(conversation_id="conv-x")
    current = SessionState()
    new = SessionState(genie=GenieState(spaces={"space_x": space}))

    merged = merge_session(current, new)

    assert len(merged.genie.spaces) == 1
    assert merged.genie.spaces["space_x"].conversation_id == "conv-x"


@pytest.mark.unit
def test_merge_session_empty_new() -> None:
    """Merging an empty new state should preserve the current state."""
    space = GenieSpaceState(conversation_id="conv-y")
    current = SessionState(genie=GenieState(spaces={"space_y": space}))
    new = SessionState()

    merged = merge_session(current, new)

    assert len(merged.genie.spaces) == 1
    assert merged.genie.spaces["space_y"].conversation_id == "conv-y"


@pytest.mark.unit
def test_merge_session_both_empty() -> None:
    """Merging two empty sessions should produce an empty session."""
    merged = merge_session(SessionState(), SessionState())

    assert len(merged.genie.spaces) == 0
