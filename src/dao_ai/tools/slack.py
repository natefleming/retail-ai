from typing import Any, Callable, Optional

from databricks.sdk.service.serving import ExternalFunctionRequestHttpMethod
from langchain_core.tools import tool
from loguru import logger
from requests import Response

from dao_ai.config import ConnectionModel


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
    # Remove '#' prefix if present
    clean_name = channel_name.lstrip("#")

    logger.debug(f"Looking up Slack channel ID for channel name: {clean_name}")

    try:
        # Call Slack API to list conversations
        response: Response = connection.workspace_client.serving_endpoints.http_request(
            conn=connection.name,
            method=ExternalFunctionRequestHttpMethod.GET,
            path="/api/conversations.list",
        )

        if response.status_code != 200:
            logger.error(f"Failed to list Slack channels: {response.text}")
            return None

        # Parse response
        data = response.json()

        if not data.get("ok"):
            logger.error(f"Slack API returned error: {data.get('error')}")
            return None

        # Search for channel by name
        channels = data.get("channels", [])
        for channel in channels:
            if channel.get("name") == clean_name:
                channel_id = channel.get("id")
                logger.debug(
                    f"Found channel ID '{channel_id}' for channel name '{clean_name}'"
                )
                return channel_id

        logger.warning(f"Channel '{clean_name}' not found in Slack workspace")
        return None

    except Exception as e:
        logger.error(f"Error looking up Slack channel: {e}")
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
    logger.debug("create_send_slack_message_tool")

    # Validate inputs
    if channel_id is None and channel_name is None:
        raise ValueError("Either channel_id or channel_name must be provided")

    # Convert connection dict to ConnectionModel if needed
    if isinstance(connection, dict):
        connection = ConnectionModel(**connection)

    # Look up channel_id from channel_name if needed
    if channel_id is None and channel_name is not None:
        logger.debug(f"Looking up channel_id for channel_name: {channel_name}")
        channel_id = _find_channel_id_by_name(connection, channel_name)
        if channel_id is None:
            raise ValueError(f"Could not find Slack channel with name '{channel_name}'")
        logger.debug(
            f"Resolved channel_name '{channel_name}' to channel_id '{channel_id}'"
        )

    if name is None:
        name = "send_slack_message"

    if description is None:
        description = "Send a message to a Slack channel"

    @tool(
        name_or_callable=name,
        description=description,
    )
    def send_slack_message(text: str) -> str:
        response: Response = connection.workspace_client.serving_endpoints.http_request(
            conn=connection.name,
            method=ExternalFunctionRequestHttpMethod.POST,
            path="/api/chat.postMessage",
            json={"channel": channel_id, "text": text},
        )

        if response.status_code == 200:
            return "Successful request sent to Slack: " + response.text
        else:
            return (
                "Encountered failure when executing request. Message from Call: "
                + response.text
            )

    return send_slack_message
