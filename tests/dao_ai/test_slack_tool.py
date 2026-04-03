"""Tests for Slack tool OBO validation."""

import pytest

from dao_ai.config import ConnectionModel
from dao_ai.tools.slack import create_send_slack_message_tool


@pytest.fixture
def mock_connection_no_obo() -> ConnectionModel:
    """Create a ConnectionModel with OBO disabled."""
    return ConnectionModel(name="slack-connection", on_behalf_of_user=False)


@pytest.fixture
def mock_connection_obo() -> ConnectionModel:
    """Create a ConnectionModel with OBO enabled."""
    return ConnectionModel(name="slack-connection", on_behalf_of_user=True)


@pytest.mark.unit
def test_slack_tool_obo_with_channel_id(mock_connection_obo: ConnectionModel) -> None:
    """When OBO is enabled and channel_id is provided, tool creation should succeed."""
    tool = create_send_slack_message_tool(
        connection=mock_connection_obo,
        channel_id="C1234567890",
    )

    assert tool is not None
    assert tool.name == "send_slack_message"


@pytest.mark.unit
def test_slack_tool_obo_with_channel_name_raises(
    mock_connection_obo: ConnectionModel,
) -> None:
    """When OBO is enabled and only channel_name is provided, should raise ValueError."""
    with pytest.raises(
        ValueError, match="channel_id is required when on_behalf_of_user is True"
    ):
        create_send_slack_message_tool(
            connection=mock_connection_obo,
            channel_name="general",
        )


@pytest.mark.unit
def test_slack_tool_no_channel_raises() -> None:
    """When neither channel_id nor channel_name is provided, should raise ValueError."""
    connection = ConnectionModel(name="slack-connection")
    with pytest.raises(
        ValueError, match="Either channel_id or channel_name must be provided"
    ):
        create_send_slack_message_tool(connection=connection)


@pytest.mark.unit
def test_slack_tool_no_obo_with_channel_id(
    mock_connection_no_obo: ConnectionModel,
) -> None:
    """When OBO is disabled and channel_id is provided, tool creation should succeed."""
    tool = create_send_slack_message_tool(
        connection=mock_connection_no_obo,
        channel_id="C1234567890",
    )

    assert tool is not None
    assert tool.name == "send_slack_message"


@pytest.mark.unit
def test_slack_tool_custom_name_and_description(
    mock_connection_obo: ConnectionModel,
) -> None:
    """Test custom name and description with OBO + channel_id."""
    tool = create_send_slack_message_tool(
        connection=mock_connection_obo,
        channel_id="C1234567890",
        name="notify_slack",
        description="Send notifications to Slack",
    )

    assert tool.name == "notify_slack"
    assert tool.description == "Send notifications to Slack"
