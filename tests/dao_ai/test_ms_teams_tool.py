"""Tests for Microsoft Teams tool validation."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk.service.serving import HttpRequestResponse

from dao_ai.config import ConnectionModel
from dao_ai.tools.ms_teams import create_send_teams_message_tool


@pytest.fixture
def mock_connection_no_obo() -> ConnectionModel:
    """Create a ConnectionModel with OBO disabled."""
    return ConnectionModel(name="teams-connection", on_behalf_of_user=False)


@pytest.fixture
def mock_connection_obo() -> ConnectionModel:
    """Create a ConnectionModel with OBO enabled."""
    return ConnectionModel(name="teams-connection", on_behalf_of_user=True)


@pytest.mark.unit
def test_teams_tool_obo_with_channel_id(mock_connection_obo: ConnectionModel) -> None:
    """When OBO is enabled and channel_id is provided, tool creation should succeed."""
    tool = create_send_teams_message_tool(
        connection=mock_connection_obo,
        team_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        channel_id="19:channel-id@thread.tacv2",
    )

    assert tool is not None
    assert tool.name == "send_teams_message"


@pytest.mark.unit
def test_teams_tool_obo_with_channel_name_raises(
    mock_connection_obo: ConnectionModel,
) -> None:
    """When OBO is enabled and only channel_name is provided, should raise ValueError."""
    with pytest.raises(
        ValueError, match="channel_id is required when on_behalf_of_user is True"
    ):
        create_send_teams_message_tool(
            connection=mock_connection_obo,
            team_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            channel_name="General",
        )


@pytest.mark.unit
def test_teams_tool_no_channel_raises() -> None:
    """When neither channel_id nor channel_name is provided, should raise ValueError."""
    connection = ConnectionModel(name="teams-connection")
    with pytest.raises(
        ValueError, match="Either channel_id or channel_name must be provided"
    ):
        create_send_teams_message_tool(
            connection=connection,
            team_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        )


@pytest.mark.unit
def test_teams_tool_no_team_id_raises() -> None:
    """When team_id is not provided, should raise ValueError."""
    connection = ConnectionModel(name="teams-connection")
    with pytest.raises(ValueError, match="team_id must be provided"):
        create_send_teams_message_tool(
            connection=connection,
            team_id="",
            channel_id="19:channel-id@thread.tacv2",
        )


@pytest.mark.unit
def test_teams_tool_no_obo_with_channel_id(
    mock_connection_no_obo: ConnectionModel,
) -> None:
    """When OBO is disabled and channel_id is provided, tool creation should succeed."""
    tool = create_send_teams_message_tool(
        connection=mock_connection_no_obo,
        team_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        channel_id="19:channel-id@thread.tacv2",
    )

    assert tool is not None
    assert tool.name == "send_teams_message"


@pytest.mark.unit
def test_teams_tool_custom_name_and_description(
    mock_connection_obo: ConnectionModel,
) -> None:
    """Test custom name and description with OBO + channel_id."""
    tool = create_send_teams_message_tool(
        connection=mock_connection_obo,
        team_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        channel_id="19:channel-id@thread.tacv2",
        name="notify_teams",
        description="Send notifications to Teams",
    )

    assert tool.name == "notify_teams"
    assert tool.description == "Send notifications to Teams"


@pytest.mark.unit
def test_teams_tool_sends_content_type_in_body() -> None:
    """The Graph API message body must include contentType field."""
    connection = ConnectionModel(
        name="teams-connection", on_behalf_of_user=True
    )
    tool = create_send_teams_message_tool(
        connection=connection,
        team_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        channel_id="19:channel-id@thread.tacv2",
    )

    mock_response = MagicMock(spec=HttpRequestResponse)
    mock_response.contents = BytesIO(b'{"id": "msg-123"}')

    mock_ws = MagicMock()
    mock_ws.serving_endpoints.http_request.return_value = mock_response

    with patch(
        "dao_ai.tools.ms_teams.ConnectionModel.workspace_client_from",
        return_value=mock_ws,
    ):
        result = tool.invoke({"text": "Hello Teams"})

    call_kwargs = mock_ws.serving_endpoints.http_request.call_args
    sent_json = json.loads(call_kwargs.kwargs["json"])

    assert sent_json["body"]["contentType"] == "text"
    assert sent_json["body"]["content"] == "Hello Teams"
    assert "Successful" in result
