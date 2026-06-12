"""Outbound WhatsApp message tool for dao-ai agents.

Mirrors :mod:`dao_ai.tools.slack` and :mod:`dao_ai.tools.ms_teams`: a
factory that returns a LangChain ``@tool`` callable. The tool POSTs to
``/{graph_api_version}/{phone_number_id}/messages`` via a Unity Catalog
HTTP Connection so the access token is managed in UC rather than as an
environment variable.

This is the **agent-initiated** path. Inbound message replies (where the
agent answers a user who messaged first) flow through
:mod:`dao_ai.apps.channels.whatsapp` and do not need this tool.

WhatsApp restricts agent-initiated outbound messages outside the 24-hour
customer service window to pre-approved templates. Plain text messages
sent via this tool will be rejected by Meta if the recipient is outside
that window — handle the resulting error in your agent prompt or wrap
this with a higher-level "send_whatsapp_template" tool if needed.
"""

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


def create_send_whatsapp_message_tool(
    connection: ConnectionModel | dict[str, Any],
    phone_number_id: str,
    graph_api_version: str = "v22.0",
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[str, str], str]:
    """Create a tool that sends a WhatsApp text message via Meta Cloud API.

    Args:
        connection: Unity Catalog connection pointing at ``graph.facebook.com``
            with a Bearer access token (Meta App / Cloud API token).
            Accepts a :class:`ConnectionModel` or its dict form.
        phone_number_id: Meta-issued phone number id (NOT the E.164 phone).
        graph_api_version: Meta Graph API version segment, e.g. ``v22.0``.
        name: Optional override for the tool name. Default ``send_whatsapp_message``.
        description: Optional override for the tool description.

    Returns:
        A LangChain tool function ``(to_wa_id, text) -> str``.
    """
    logger.trace("Creating send WhatsApp message tool")

    if isinstance(connection, dict):
        connection = ConnectionModel(**connection)

    if not phone_number_id:
        raise ValueError("phone_number_id is required")

    if name is None:
        name = "send_whatsapp_message"

    if description is None:
        description = (
            "Send a WhatsApp text message to a recipient identified by their "
            "phone number in E.164 format (no '+' prefix; e.g. '14155552671'). "
            "Subject to Meta's 24-hour customer service window — outside that "
            "window only pre-approved templates can be sent and this call "
            "will return an error."
        )

    outbound_path = f"/{graph_api_version}/{phone_number_id}/messages"

    @tool(name_or_callable=name, description=description)
    def send_whatsapp_message(
        to_wa_id: str,
        text: str,
        runtime: ToolRuntime[Context] = None,
    ) -> str:
        """Send a single WhatsApp text message.

        Args:
            to_wa_id: Recipient phone number in E.164 (no '+'), e.g. '14155552671'.
            text: Message body. Will be sent verbatim with link previews disabled.
        """
        from databricks.sdk import WorkspaceClient

        set_resource_attributes(
            ResourceInfo("whatsapp", connection.on_behalf_of_user, connection.name)
        )

        context: Context | None = runtime.context if runtime else None
        workspace_client: WorkspaceClient = connection.workspace_client_from(context)

        body = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_wa_id,
            "type": "text",
            "text": {"body": text, "preview_url": False},
        }

        try:
            response: HttpRequestResponse = (
                workspace_client.serving_endpoints.http_request(
                    connection_name=connection.name,
                    method=ExternalFunctionRequestHttpMethod.POST,
                    path=outbound_path,
                    json=json_module.dumps(body),
                )
            )
            body_text: str = _read_response_body(response)
            return "Successful request sent to WhatsApp: " + body_text
        except Exception as e:
            return (
                "Encountered failure when executing request. Message from Call: "
                + str(e)
            )

    return send_whatsapp_message


__all__ = ["create_send_whatsapp_message_tool"]
