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
    logger.info("=== Creating send_email_tool ===")
    logger.debug(
        f"Factory called with config type: {type(smtp_config).__name__}, "
        f"name={name}, description={description}"
    )

    # Convert dict to SMTPConfigModel if needed
    if isinstance(smtp_config, dict):
        logger.debug("Converting dict config to SMTPConfigModel")
        smtp_config = SMTPConfigModel(**smtp_config)
    else:
        logger.debug("Config already is SMTPConfigModel")

    # Resolve all variable values
    logger.debug("Resolving SMTP configuration variables...")

    logger.debug("  - Resolving host")
    host: str = value_of(smtp_config.host)
    logger.debug(f"    Host resolved: {host}")

    logger.debug("  - Resolving port")
    port: int = int(value_of(smtp_config.port))
    logger.debug(f"    Port resolved: {port}")

    logger.debug("  - Resolving username")
    username: str = value_of(smtp_config.username)
    logger.debug(f"    Username resolved: {username}")

    logger.debug("  - Resolving password")
    password: str = value_of(smtp_config.password)
    logger.debug(
        f"    Password resolved: {'*' * len(password) if password else 'None'}"
    )

    logger.debug("  - Resolving sender_email")
    sender_email: str = (
        value_of(smtp_config.sender_email) if smtp_config.sender_email else username
    )
    logger.debug(
        f"    Sender email resolved: {sender_email} "
        f"({'from sender_email' if smtp_config.sender_email else 'defaulted to username'})"
    )

    use_tls: bool = smtp_config.use_tls
    logger.debug(f"  - TLS enabled: {use_tls}")

    logger.info(
        f"SMTP configuration resolved - host={host}, port={port}, "
        f"sender={sender_email}, use_tls={use_tls}"
    )

    if name is None:
        name = "send_email"
        logger.debug(f"Tool name defaulted to: {name}")
    else:
        logger.debug(f"Tool name set to: {name}")

    if description is None:
        description = "Send an email to a recipient with subject and body content"
        logger.debug("Tool description using default")
    else:
        logger.debug(f"Tool description set to: {description}")

    logger.info(f"Creating tool '{name}' with @tool decorator")

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
        logger.info("=== send_email tool invoked ===")
        logger.info(f"  To: {to}")
        logger.info(f"  Subject: {subject}")
        logger.info(f"  Body length: {len(body)} characters")
        logger.info(f"  CC: {cc if cc else 'None'}")

        try:
            logger.debug("Constructing email message...")

            # Create message
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = to
            msg["Subject"] = subject
            logger.debug(f"  From: {sender_email}")
            logger.debug(f"  To: {to}")
            logger.debug(f"  Subject: {subject}")

            if cc:
                msg["Cc"] = cc
                logger.debug(f"  CC: {cc}")

            # Attach body as plain text
            msg.attach(MIMEText(body, "plain"))
            logger.debug(f"  Body attached ({len(body)} chars)")

            # Send email
            logger.info(f"Connecting to SMTP server {host}:{port}...")
            with smtplib.SMTP(host, port) as server:
                logger.debug("SMTP connection established")

                if use_tls:
                    logger.debug("Upgrading connection to TLS...")
                    server.starttls()
                    logger.debug("TLS upgrade successful")

                logger.debug(f"Authenticating with username: {username}")
                server.login(username, password)
                logger.info("SMTP authentication successful")

                # Build recipient list
                recipients = [to]
                if cc:
                    cc_addresses = [addr.strip() for addr in cc.split(",")]
                    recipients.extend(cc_addresses)
                    logger.debug(f"Total recipients: {len(recipients)} ({recipients})")
                else:
                    logger.debug(f"Single recipient: {to}")

                logger.info(f"Sending message to {len(recipients)} recipient(s)...")
                server.send_message(msg)
                logger.info("Message sent successfully via SMTP")

            success_msg = f"✓ Email sent successfully to {to}"
            if cc:
                success_msg += f" (cc: {cc})"

            logger.info(success_msg)
            logger.info("=== send_email completed successfully ===")
            return success_msg

        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"✗ SMTP authentication failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"  Server: {host}:{port}")
            logger.error(f"  Username: {username}")
            logger.error("=== send_email failed (authentication) ===")
            return error_msg
        except smtplib.SMTPException as e:
            error_msg = f"✗ SMTP error: {str(e)}"
            logger.error(error_msg)
            logger.error(f"  Server: {host}:{port}")
            logger.error("=== send_email failed (SMTP error) ===")
            return error_msg
        except Exception as e:
            error_msg = f"✗ Failed to send email: {str(e)}"
            logger.error(error_msg)
            logger.error(f"  Error type: {type(e).__name__}")
            logger.error("=== send_email failed (unexpected error) ===")
            return error_msg

    logger.info(f"Tool '{name}' created successfully")
    logger.info("=== send_email_tool creation complete ===")
    return send_email
