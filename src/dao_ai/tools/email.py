"""Email tool for sending emails via SMTP."""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable, Optional

from langchain_core.tools import tool
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from dao_ai.config import AnyVariable, value_of


class SMTPConfigModel(BaseModel):
    """Configuration model for SMTP email settings."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    host: AnyVariable = Field(
        default="smtp.gmail.com",
        description="SMTP server hostname",
    )
    port: AnyVariable = Field(
        default=587,
        description="SMTP server port",
    )
    username: AnyVariable = Field(
        description="SMTP username for authentication",
    )
    password: AnyVariable = Field(
        description="SMTP password for authentication",
    )
    sender_email: Optional[AnyVariable] = Field(
        default=None,
        description="Email address to use as sender (defaults to username)",
    )
    use_tls: bool = Field(
        default=True,
        description="Whether to use TLS encryption",
    )


def create_send_email_tool(
    smtp_config: SMTPConfigModel | dict[str, Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[str, str, str, Optional[str]], str]:
    """
    Create a tool that sends emails via SMTP.

    This factory function creates a tool for sending emails with configurable SMTP settings.
    All configuration values support AnyVariable types, allowing use of environment variables,
    secrets, and composite variables.

    Args:
        smtp_config: SMTP configuration (SMTPConfigModel or dict). Supports:
            - host: SMTP server hostname (supports variables/secrets)
            - port: SMTP server port (supports variables/secrets)
            - username: SMTP username (supports variables/secrets)
            - password: SMTP password (supports variables/secrets)
            - sender_email: Sender email address, defaults to username (supports variables/secrets)
            - use_tls: Whether to use TLS encryption (default: True)
        name: Custom tool name (default: 'send_email')
        description: Custom tool description

    Returns:
        A tool function that sends emails via SMTP

    Example:
        Basic usage with environment variables:
        ```yaml
        tools:
          send_email:
            name: send_email
            function:
              type: factory
              name: dao_ai.tools.email.create_send_email_tool
              args:
                smtp_config:
                  host: smtp.gmail.com
                  port: 587
                  username: ${SMTP_USER}
                  password: ${SMTP_PASSWORD}
                  sender_email: bot@example.com
        ```

        With secrets:
        ```yaml
        tools:
          send_email:
            name: send_email
            function:
              type: factory
              name: dao_ai.tools.email.create_send_email_tool
              args:
                smtp_config:
                  host: smtp.gmail.com
                  port: 587
                  username:
                    type: secret
                    scope: email
                    key: smtp_user
                  password:
                    type: secret
                    scope: email
                    key: smtp_password
        ```
    """
    logger.debug(
        "Creating send_email_tool",
        config_type=type(smtp_config).__name__,
        tool_name=name,
    )

    # Convert dict to SMTPConfigModel if needed
    if isinstance(smtp_config, dict):
        smtp_config = SMTPConfigModel(**smtp_config)

    # Resolve all variable values
    host: str = value_of(smtp_config.host)
    port: int = int(value_of(smtp_config.port))
    username: str = value_of(smtp_config.username)
    password: str = value_of(smtp_config.password)
    sender_email: str = (
        value_of(smtp_config.sender_email) if smtp_config.sender_email else username
    )
    use_tls: bool = smtp_config.use_tls

    logger.info(
        "SMTP configuration resolved",
        host=host,
        port=port,
        sender=sender_email,
        use_tls=use_tls,
        password_set=bool(password),
    )

    if name is None:
        name = "send_email"
    if description is None:
        description = "Send an email to a recipient with subject and body content"

    logger.debug("Creating email tool with decorator", tool_name=name)

    @tool(
        name_or_callable=name,
        description=description,
    )
    def send_email(
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
    ) -> str:
        """
        Send an email via SMTP.

        Args:
            to: Recipient email address
            subject: Email subject line
            body: Email body content (plain text)
            cc: Optional CC recipients (comma-separated email addresses)

        Returns:
            str: Success or error message
        """
        logger.info(
            "Sending email", to=to, subject=subject, body_length=len(body), cc=cc
        )

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = to
            msg["Subject"] = subject

            if cc:
                msg["Cc"] = cc

            # Attach body as plain text
            msg.attach(MIMEText(body, "plain"))

            # Send email
            logger.debug("Connecting to SMTP server", host=host, port=port)
            with smtplib.SMTP(host, port) as server:
                if use_tls:
                    logger.trace("Upgrading to TLS")
                    server.starttls()

                logger.trace("Authenticating", username=username)
                server.login(username, password)

                # Build recipient list
                recipients = [to]
                if cc:
                    cc_addresses = [addr.strip() for addr in cc.split(",")]
                    recipients.extend(cc_addresses)

                logger.debug("Sending message", recipients_count=len(recipients))
                server.send_message(msg)

            success_msg = f"✓ Email sent successfully to {to}"
            if cc:
                success_msg += f" (cc: {cc})"

            logger.success("Email sent successfully", to=to, cc=cc)
            return success_msg

        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"✗ SMTP authentication failed: {str(e)}"
            logger.error(
                "SMTP authentication failed",
                server=f"{host}:{port}",
                username=username,
                error=str(e),
            )
            return error_msg
        except smtplib.SMTPException as e:
            error_msg = f"✗ SMTP error: {str(e)}"
            logger.error("SMTP error", server=f"{host}:{port}", error=str(e))
            return error_msg
        except Exception as e:
            error_msg = f"✗ Failed to send email: {str(e)}"
            logger.error(
                "Failed to send email", error_type=type(e).__name__, error=str(e)
            )
            return error_msg

    logger.success("Email tool created", tool_name=name)
    return send_email
