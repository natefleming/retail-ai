import json as json_module
from typing import Any, Callable, Optional

from databricks.sdk.service.serving import (
    ExternalFunctionRequestHttpMethod,
    HttpRequestResponse,
)
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from loguru import logger

from dao_ai.config import ConnectionModel
from dao_ai.state import Context
from dao_ai.tools.rest_api import _read_response_body
from dao_ai.tools.tracing import ResourceInfo, set_resource_attributes


def _find_channel_id_by_name(
    connection: ConnectionModel, team_id: str, channel_name: str
) -> Optional[str]:
    """
    Find a Microsoft Teams channel ID by channel name using the Graph API.

    Args:
        connection: ConnectionModel with workspace_client
        team_id: Microsoft Teams team ID (GUID)
        channel_name: Name of the Teams channel (e.g., 'General')

    Returns:
        Channel ID if found, None otherwise
    """
    clean_name = channel_name.strip()

    logger.trace(
        "Looking up Teams channel ID", channel_name=clean_name, team_id=team_id
    )

    try:
        response: HttpRequestResponse = (
            connection.workspace_client.serving_endpoints.http_request(
                connection_name=connection.name,
                method=ExternalFunctionRequestHttpMethod.GET,
                path=f"/teams/{team_id}/channels",
            )
        )

        body_text: str = _read_response_body(response)
        data: dict[str, Any] = json_module.loads(body_text) if body_text else {}

        if "error" in data:
            logger.error(
                "Microsoft Graph API returned error",
                error=data["error"].get("message", str(data["error"])),
            )
            return None

        channels = data.get("value", [])
        for channel in channels:
            if channel.get("displayName") == clean_name:
                channel_id = channel.get("id")
                logger.debug(
                    "Found Teams channel ID",
                    channel_id=channel_id,
                    channel_name=clean_name,
                )
                return channel_id

        logger.warning("Teams channel not found", channel_name=clean_name)
        return None

    except Exception as e:
        logger.error(
            "Error looking up Teams channel",
            channel_name=clean_name,
            team_id=team_id,
            error=str(e),
        )
        return None


def create_send_teams_message_tool(
    connection: ConnectionModel | dict[str, Any],
    team_id: str,
    channel_id: Optional[str] = None,
    channel_name: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[str], str]:
    """
    Create a tool that sends a message to a Microsoft Teams channel.

    Args:
        connection: Unity Catalog connection to Microsoft Graph API (ConnectionModel or dict)
        team_id: Microsoft Teams team ID (GUID)
        channel_id: Teams channel ID. If not provided, channel_name is used.
        channel_name: Teams channel name (e.g., 'General'). Used to lookup channel_id if not provided.
        name: Custom tool name (default: 'send_teams_message')
        description: Custom tool description

    Returns:
        A tool function that sends messages to the specified Teams channel
    """
    logger.trace("Creating send Teams message tool")

    if not team_id:
        raise ValueError("team_id must be provided")

    if channel_id is None and channel_name is None:
        raise ValueError("Either channel_id or channel_name must be provided")

    if isinstance(connection, dict):
        connection = ConnectionModel(**connection)

    if channel_id is None and channel_name is not None:
        if connection.on_behalf_of_user:
            raise ValueError(
                "channel_id is required when on_behalf_of_user is True. "
                "Name-based channel lookup cannot authenticate at startup."
            )
        logger.trace(
            "Looking up channel ID for channel name", channel_name=channel_name
        )
        channel_id = _find_channel_id_by_name(connection, team_id, channel_name)
        if channel_id is None:
            raise ValueError(f"Could not find Teams channel with name '{channel_name}'")
        logger.debug(
            "Resolved channel name to ID",
            channel_name=channel_name,
            channel_id=channel_id,
        )

    if name is None:
        name = "send_teams_message"

    if description is None:
        description = "Send a message to a Microsoft Teams channel"

    @tool(
        name_or_callable=name,
        description=description,
    )
    def send_teams_message(
        text: str,
        runtime: ToolRuntime[Context] = None,
    ) -> str:
        from databricks.sdk import WorkspaceClient

        set_resource_attributes(
            ResourceInfo("ms_teams", connection.on_behalf_of_user, connection.name)
        )

        context: Context | None = runtime.context if runtime else None
        workspace_client: WorkspaceClient = connection.workspace_client_from(context)

        try:
            response: HttpRequestResponse = (
                workspace_client.serving_endpoints.http_request(
                    connection_name=connection.name,
                    method=ExternalFunctionRequestHttpMethod.POST,
                    path=f"/teams/{team_id}/channels/{channel_id}/messages",
                    json=json_module.dumps({"body": {"contentType": "text", "content": text}}),
                )
            )
            body_text: str = _read_response_body(response)
            return "Successful request sent to Microsoft Teams: " + body_text
        except Exception as e:
            return (
                "Encountered failure when executing request. Message from Call: "
                + str(e)
            )

    return send_teams_message
