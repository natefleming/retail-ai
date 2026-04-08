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
    connection: ConnectionModel, channel_name: str
) -> Optional[str]:
    """
    Find a Slack channel ID by channel name using the conversations.list API.

    Based on: https://docs.databricks.com/aws/en/generative-ai/agent-framework/slack-agent

    Args:
        connection: ConnectionModel with workspace_client
        channel_name: Name of the Slack channel (with or without '#' prefix)

    Returns:
        Channel ID if found, None otherwise
    """
    clean_name = channel_name.lstrip("#")

    logger.trace("Looking up Slack channel ID", channel_name=clean_name)

    try:
        response: HttpRequestResponse = (
            connection.workspace_client.serving_endpoints.http_request(
                connection_name=connection.name,
                method=ExternalFunctionRequestHttpMethod.GET,
                path="/api/conversations.list",
            )
        )

        body_text: str = _read_response_body(response)
        data: dict[str, Any] = json_module.loads(body_text) if body_text else {}

        if not data.get("ok"):
            logger.error("Slack API returned error", error=data.get("error"))
            return None

        channels = data.get("channels", [])
        for channel in channels:
            if channel.get("name") == clean_name:
                channel_id = channel.get("id")
                logger.debug(
                    "Found Slack channel ID",
                    channel_id=channel_id,
                    channel_name=clean_name,
                )
                return channel_id

        logger.warning("Slack channel not found", channel_name=clean_name)
        return None

    except Exception as e:
        logger.error(
            "Error looking up Slack channel", channel_name=clean_name, error=str(e)
        )
        return None


def create_send_slack_message_tool(
    connection: ConnectionModel | dict[str, Any],
    channel_id: Optional[str] = None,
    channel_name: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[str], str]:
    """
    Create a tool that sends a message to a Slack channel.

    Args:
        connection: Unity Catalog connection to Slack (ConnectionModel or dict)
        channel_id: Slack channel ID (e.g., 'C1234567890'). If not provided, channel_name is used.
        channel_name: Slack channel name (e.g., 'general' or '#general'). Used to lookup channel_id if not provided.
        name: Custom tool name (default: 'send_slack_message')
        description: Custom tool description

    Returns:
        A tool function that sends messages to the specified Slack channel

    Based on: https://docs.databricks.com/aws/en/generative-ai/agent-framework/slack-agent
    """
    logger.trace("Creating send Slack message tool")

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
        channel_id = _find_channel_id_by_name(connection, channel_name)
        if channel_id is None:
            raise ValueError(f"Could not find Slack channel with name '{channel_name}'")
        logger.debug(
            "Resolved channel name to ID",
            channel_name=channel_name,
            channel_id=channel_id,
        )

    if name is None:
        name = "send_slack_message"

    if description is None:
        description = "Send a message to a Slack channel"

    @tool(
        name_or_callable=name,
        description=description,
    )
    def send_slack_message(
        text: str,
        runtime: ToolRuntime[Context] = None,
    ) -> str:
        from databricks.sdk import WorkspaceClient

        set_resource_attributes(
            ResourceInfo("slack", connection.on_behalf_of_user, connection.name)
        )

        context: Context | None = runtime.context if runtime else None
        workspace_client: WorkspaceClient = connection.workspace_client_from(context)

        try:
            response: HttpRequestResponse = (
                workspace_client.serving_endpoints.http_request(
                    connection_name=connection.name,
                    method=ExternalFunctionRequestHttpMethod.POST,
                    path="/api/chat.postMessage",
                    json=json_module.dumps({"channel": channel_id, "text": text}),
                )
            )
            body_text: str = _read_response_body(response)
            return "Successful request sent to Slack: " + body_text
        except Exception as e:
            return (
                "Encountered failure when executing request. Message from Call: "
                + str(e)
            )

    return send_slack_message
