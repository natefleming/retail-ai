import argparse
import getpass
import json
import os
import subprocess
import sys
import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional, Sequence

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from dao_ai.config import AppConfig
from dao_ai.graph import create_dao_ai_graph
from dao_ai.logging import configure_logging
from dao_ai.models import save_image
from dao_ai.utils import normalize_name

configure_logging(level="ERROR")


def get_default_user_id() -> str:
    """
    Get the default user ID for the CLI session.

    Tries to get the current user from Databricks, falls back to local user.

    Returns:
        User ID string (Databricks username or local username)
    """
    try:
        # Try to get current user from Databricks SDK
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        current_user = w.current_user.me()
        user_id = current_user.user_name
        logger.debug(f"Using Databricks user: {user_id}")
        return user_id
    except Exception as e:
        # Fall back to local system user
        logger.debug(f"Could not get Databricks user, using local user: {e}")
        local_user = getpass.getuser()
        logger.debug(f"Using local user: {local_user}")
        return local_user


def detect_cloud_provider(profile: Optional[str] = None) -> Optional[str]:
    """
    Detect the cloud provider from the Databricks workspace URL.

    The cloud provider is determined by the workspace URL pattern:
    - Azure: *.azuredatabricks.net
    - AWS: *.cloud.databricks.com (without gcp subdomain)
    - GCP: *.gcp.databricks.com

    Args:
        profile: Optional Databricks CLI profile name

    Returns:
        Cloud provider string ('azure', 'aws', 'gcp') or None if detection fails
    """
    try:
        import os

        from databricks.sdk import WorkspaceClient

        # Check for environment variables that might override profile
        if profile and os.environ.get("DATABRICKS_HOST"):
            logger.warning(
                f"DATABRICKS_HOST environment variable is set, which may override --profile {profile}"
            )

        # Create workspace client with optional profile
        if profile:
            logger.debug(f"Creating WorkspaceClient with profile: {profile}")
            w = WorkspaceClient(profile=profile)
        else:
            logger.debug("Creating WorkspaceClient with default/ambient credentials")
            w = WorkspaceClient()

        # Get the workspace URL from config
        host = w.config.host
        logger.debug(f"WorkspaceClient host: {host}, profile used: {profile}")
        if not host:
            logger.warning("Could not determine workspace URL for cloud detection")
            return None

        host_lower = host.lower()

        if "azuredatabricks.net" in host_lower:
            logger.debug(f"Detected Azure cloud from workspace URL: {host}")
            return "azure"
        elif ".gcp.databricks.com" in host_lower:
            logger.debug(f"Detected GCP cloud from workspace URL: {host}")
            return "gcp"
        elif ".cloud.databricks.com" in host_lower or "databricks.com" in host_lower:
            # AWS uses *.cloud.databricks.com or regional patterns
            logger.debug(f"Detected AWS cloud from workspace URL: {host}")
            return "aws"
        else:
            logger.warning(f"Could not determine cloud provider from URL: {host}")
            return None

    except Exception as e:
        logger.warning(f"Could not detect cloud provider: {e}")
        return None


env_path: str = find_dotenv()
if env_path:
    logger.info(f"Loading environment variables from: {env_path}")
    _ = load_dotenv(env_path)


def parse_args(args: Sequence[str]) -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog="dao-ai",
        description="DAO AI Agent Command Line Interface - A comprehensive tool for managing, validating, and visualizing multi-agent DAO AI systems",
        epilog="""
Examples:
  dao-ai schema                                          # Generate JSON schema for configuration validation
  dao-ai validate -c config/model_config.yaml            # Validate a specific configuration file
  dao-ai graph -o architecture.png -c my_config.yaml -v  # Generate visual graph with verbose output
  dao-ai chat -c config/retail.yaml --custom-input store_num=87887  # Start interactive chat session
  dao-ai list-mcp-tools -c config/mcp_config.yaml --apply-filters  # List filtered MCP tools only
  dao-ai validate                                        # Validate with detailed logging
  dao-ai bundle --deploy                                 # Deploy the DAO AI asset bundle
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (use -v, -vv, -vvv, or -vvvv for ERROR, WARNING, INFO, DEBUG, or TRACE levels)",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands for managing the DAO AI system",
        metavar="COMMAND",
    )

    # Schema command
    _: ArgumentParser = subparsers.add_parser(
        "schema",
        help="Generate JSON schema for configuration validation",
        description="""
Generate the JSON schema definition for the DAO AI configuration format.
This schema can be used for IDE autocompletion, validation tools, and documentation.
The output is a complete JSON Schema that describes all valid configuration options,
including agents, tools, models, orchestration patterns, and guardrails.
        """,
        epilog="Example: dao-ai schema > config_schema.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Validation command
    validation_parser: ArgumentParser = subparsers.add_parser(
        "validate",
        help="Validate configuration file syntax and semantics",
        description="""
Validate a DAO AI configuration file for correctness and completeness.
This command checks:
- YAML syntax and structure
- Required fields and data types
- Agent configurations and dependencies
- Tool definitions and availability
- Model specifications and compatibility
- Orchestration patterns (supervisor/swarm)
- Guardrail configurations

Exit codes:
  0 - Configuration is valid
  1 - Configuration contains errors
        """,
        epilog="""
Examples:
  dao-ai validate                                  # Validate default ./config/model_config.yaml
  dao-ai validate -c config/production.yaml       # Validate specific config file
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validation_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file to validate (default: ./config/model_config.yaml)",
    )

    # Graph command
    graph_parser: ArgumentParser = subparsers.add_parser(
        "graph",
        help="Generate visual representation of the agent workflow",
        description="""
Generate a visual graph representation of the configured DAO AI system.
This creates a diagram showing:
- Agent nodes and their relationships
- Orchestration flow (supervisor or swarm patterns)
- Tool dependencies and connections
- Message routing and state transitions
- Conditional logic and decision points
        """,
        epilog="""
Examples:
  dao-ai graph -o architecture.png                # Generate PNG diagram
  dao-ai graph -o workflow.png -c prod.yaml       # Generate PNG from specific config
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    graph_parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        metavar="FILE",
        help="Output file path for the generated graph.",
    )
    graph_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file to visualize",
    )

    bundle_parser: ArgumentParser = subparsers.add_parser(
        "bundle",
        help="Bundle configuration for deployment",
        description="""
Perform operations on the DAO AI asset bundle.
This command prepares the configuration for deployment by:
- Deploying DAO AI as an asset bundle
- Running the DAO AI system with the current configuration
""",
        epilog="""
Examples:
    dao-ai bundle --deploy
    dao-ai bundle --run
""",
    )

    bundle_parser.add_argument(
        "-p",
        "--profile",
        type=str,
        help="The Databricks profile to use for deployment",
    )
    bundle_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file for the bundle",
    )
    bundle_parser.add_argument(
        "-d",
        "--deploy",
        action="store_true",
        help="Deploy the DAO AI asset bundle",
    )
    bundle_parser.add_argument(
        "--destroy",
        action="store_true",
        help="Destroy the DAO AI asset bundle",
    )
    bundle_parser.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="Run the DAO AI system with the current configuration",
    )
    bundle_parser.add_argument(
        "-t",
        "--target",
        type=str,
        help="Bundle target name (default: auto-generated from app name and cloud)",
    )
    bundle_parser.add_argument(
        "--cloud",
        type=str,
        choices=["azure", "aws", "gcp"],
        help="Cloud provider (auto-detected from workspace URL if not specified)",
    )
    bundle_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without executing the deployment or run commands",
    )
    bundle_parser.add_argument(
        "--deployment-target",
        type=str,
        choices=["model_serving", "apps"],
        default=None,
        help="Agent deployment target: 'model_serving' or 'apps'. "
        "If not specified, uses app.deployment_target from config file, "
        "or defaults to 'model_serving'. Passed to the deploy notebook.",
    )

    # Deploy command
    deploy_parser: ArgumentParser = subparsers.add_parser(
        "deploy",
        help="Deploy configuration file syntax and semantics",
        description="""
Deploy the DAO AI system using the specified configuration file.
This command validates the configuration and deploys the DAO AI agents, tools, and models to the
        """,
        epilog="""
Examples:
  dao-ai deploy                                  # Validate default ./config/model_config.yaml
  dao-ai deploy -c config/production.yaml       # Validate specific config file
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    deploy_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file to validate",
    )
    deploy_parser.add_argument(
        "-t",
        "--target",
        type=str,
        choices=["model_serving", "apps"],
        default=None,
        help="Deployment target: 'model_serving' or 'apps'. "
        "If not specified, uses app.deployment_target from config file, "
        "or defaults to 'model_serving'.",
    )

    # List MCP tools command
    list_mcp_parser: ArgumentParser = subparsers.add_parser(
        "list-mcp-tools",
        help="List available MCP tools from configuration",
        description="""
List all available MCP tools from the configured MCP servers.
This command shows:
- All MCP servers/functions in the configuration
- Available tools from each server
- Full descriptions for each tool (no truncation)
- Tool parameters in readable format (type, required/optional, descriptions)
- Which tools are included/excluded based on filters
- Filter patterns (include_tools, exclude_tools)

Use this command to:
- Discover available tools before configuring agents
- Review tool descriptions and parameter schemas
- Debug tool filtering configuration
- Verify MCP server connectivity

Options:
- Use --apply-filters to only show tools that will be loaded (hides excluded tools)
- Without --apply-filters, see all available tools with include/exclude status

Note: Schemas are displayed in a concise, readable format instead of verbose JSON
        """,
        epilog="""Examples:
  dao-ai list-mcp-tools -c config/model_config.yaml
  dao-ai list-mcp-tools -c config/model_config.yaml --apply-filters
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    list_mcp_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/model_config.yaml",
        required=False,
        metavar="FILE",
        help="Path to the model configuration file (default: ./config/model_config.yaml)",
    )
    list_mcp_parser.add_argument(
        "--apply-filters",
        action="store_true",
        help="Only show tools that pass include/exclude filters (hide excluded tools)",
    )

    chat_parser: ArgumentParser = subparsers.add_parser(
        "chat",
        help="Interactive chat with the DAO AI system",
        description="""
Start an interactive chat session with the DAO AI system.
This command provides a REPL (Read-Eval-Print Loop) interface where you can
send messages to the configured agents and receive streaming responses in real-time.

The chat session maintains conversation history and supports the full agent
orchestration capabilities defined in your configuration file.

Use Ctrl-D (EOF) to exit the chat session gracefully.
Use Ctrl-C to interrupt and exit immediately.
        """,
        epilog="""
Examples:
  dao-ai chat -c config/model_config.yaml                              # Start chat (auto-detects user)
  dao-ai chat -c config/retail.yaml --custom-input store_num=87887     # Chat with custom store number
  dao-ai chat -c config/prod.yaml --user-id john.doe@company.com       # Chat with specific user ID
  dao-ai chat -c config/retail.yaml --custom-input store_num=123 --custom-input region=west  # Multiple custom inputs
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    chat_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file to validate",
    )
    chat_parser.add_argument(
        "--custom-input",
        action="append",
        metavar="KEY=VALUE",
        help="Custom configurable input as key=value pair (can be used multiple times)",
    )
    chat_parser.add_argument(
        "--user-id",
        type=str,
        default=None,  # Will be set to actual user in handle_chat_command
        metavar="ID",
        help="User ID for the chat session (default: current Databricks user or local username)",
    )
    chat_parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        metavar="ID",
        help="Thread ID for the chat session (default: auto-generated UUID)",
    )

    options = parser.parse_args(args)

    # Generate a new thread_id UUID if not provided (only for chat command)
    if hasattr(options, "thread_id") and options.thread_id is None:
        import uuid

        options.thread_id = str(uuid.uuid4())

    return options


def handle_chat_command(options: Namespace) -> None:
    """Interactive chat REPL with the DAO AI system with Human-in-the-Loop support."""
    logger.debug("Starting chat session with DAO AI system...")

    try:
        # Set default user_id if not provided
        if options.user_id is None:
            options.user_id = get_default_user_id()

        config: AppConfig = AppConfig.from_file(options.config)
        app = create_dao_ai_graph(config)

        print("🤖 DAO AI Chat Session Started")
        print("Type your message and press Enter. Use Ctrl-D to exit.")
        print("-" * 50)

        # Show current configuration
        print("📋 Session Configuration:")
        print(f"   Config file: {options.config}")
        print(f"   Thread ID: {options.thread_id}")
        print(f"   User ID: {options.user_id}")
        if options.custom_input:
            print("   Custom inputs:")
            for custom_input in options.custom_input:
                print(f"     {custom_input}")
        print("-" * 50)

        # Import streaming function and interrupt handling
        from langchain_core.messages import AIMessage, HumanMessage

        # Conversation history
        messages = []

        while True:
            try:
                # Read user input
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                # Add user message to history
                user_message = HumanMessage(content=user_input)
                messages.append(user_message)

                # Parse custom inputs from command line
                configurable = {
                    "thread_id": options.thread_id,
                    "user_id": options.user_id,
                }

                # Add custom key=value pairs if provided
                if options.custom_input:
                    for custom_input in options.custom_input:
                        try:
                            key, value = custom_input.split("=", 1)
                            # Try to convert to appropriate type
                            if value.isdigit():
                                configurable[key] = int(value)
                            elif value.lower() in ("true", "false"):
                                configurable[key] = value.lower() == "true"
                            elif value.replace(".", "", 1).isdigit():
                                configurable[key] = float(value)
                            else:
                                configurable[key] = value
                        except ValueError:
                            print(
                                f"⚠️  Warning: Invalid custom input format '{custom_input}'. Expected key=value format."
                            )
                            continue

                # Normalize user_id for memory namespace compatibility (replace . with _)
                # This matches the normalization in models.py _convert_to_context
                if configurable.get("user_id"):
                    configurable["user_id"] = configurable["user_id"].replace(".", "_")

                # Create Context object from configurable dict
                from dao_ai.state import Context

                context = Context(**configurable)

                # Prepare config with all context fields for checkpointer/memory
                # Note: langmem tools require user_id in config.configurable for namespace resolution
                config = {"configurable": context.model_dump()}

                # Invoke the graph and handle interrupts (HITL)
                # Wrap in async function to maintain connection pool throughout
                logger.debug(f"Invoking graph with {len(messages)} messages")
                logger.debug(f"Context: {context}")
                logger.debug(f"Config: {config}")

                import asyncio

                from langgraph.types import Command

                async def _invoke_with_hitl():
                    """Invoke graph and handle HITL interrupts in single async context."""
                    result = await app.ainvoke(
                        {"messages": messages},
                        config=config,
                        context=context,  # Pass context as separate parameter
                    )

                    # Check for interrupts (Human-in-the-Loop) using __interrupt__
                    # This is the modern LangChain pattern
                    while "__interrupt__" in result:
                        interrupts = result["__interrupt__"]
                        logger.info(f"HITL: {len(interrupts)} interrupt(s) detected")

                        # Collect decisions for all interrupts
                        decisions = []

                        for interrupt in interrupts:
                            interrupt_value = interrupt.value
                            action_requests = interrupt_value.get("action_requests", [])

                            for action_request in action_requests:
                                # Display interrupt information
                                print("\n⚠️  Human in the Loop - Tool Approval Required")
                                print(f"{'=' * 60}")

                                tool_name = action_request.get("name", "unknown")
                                tool_args = action_request.get("arguments", {})
                                description = action_request.get("description", "")

                                print(f"Tool: {tool_name}")
                                if description:
                                    print(f"\n{description}\n")

                                print("Arguments:")
                                for arg_name, arg_value in tool_args.items():
                                    # Truncate long values
                                    arg_str = str(arg_value)
                                    if len(arg_str) > 100:
                                        arg_str = arg_str[:97] + "..."
                                    print(f"  - {arg_name}: {arg_str}")

                                print(f"{'=' * 60}")

                                # Prompt user for decision
                                while True:
                                    decision_input = (
                                        input(
                                            "\nAction? (a)pprove / (r)eject / (e)dit / (h)elp: "
                                        )
                                        .strip()
                                        .lower()
                                    )

                                    if decision_input in ["a", "approve"]:
                                        logger.info("User approved tool call")
                                        print("✅ Approved - continuing execution...")
                                        decisions.append({"type": "approve"})
                                        break
                                    elif decision_input in ["r", "reject"]:
                                        logger.info("User rejected tool call")
                                        feedback = input(
                                            "   Feedback for agent (optional): "
                                        ).strip()
                                        if feedback:
                                            decisions.append(
                                                {"type": "reject", "message": feedback}
                                            )
                                        else:
                                            decisions.append(
                                                {
                                                    "type": "reject",
                                                    "message": "Tool call rejected by user",
                                                }
                                            )
                                        print(
                                            "❌ Rejected - agent will receive feedback..."
                                        )
                                        break
                                    elif decision_input in ["e", "edit"]:
                                        print(
                                            "ℹ️  Edit functionality not yet implemented in CLI"
                                        )
                                        print("   Please approve or reject.")
                                        continue
                                    elif decision_input in ["h", "help"]:
                                        print("\nAvailable actions:")
                                        print(
                                            "  (a)pprove - Execute the tool call as shown"
                                        )
                                        print(
                                            "  (r)eject  - Cancel the tool call with optional feedback"
                                        )
                                        print(
                                            "  (e)dit    - Modify arguments (not yet implemented)"
                                        )
                                        print("  (h)elp    - Show this help message")
                                        continue
                                    else:
                                        print("Invalid option. Type 'h' for help.")
                                        continue

                        # Resume execution with decisions using Command
                        # This is the modern LangChain pattern
                        logger.debug(f"Resuming with {len(decisions)} decision(s)")
                        result = await app.ainvoke(
                            Command(resume={"decisions": decisions}),
                            config=config,
                            context=context,
                        )

                    return result

                try:
                    # Use async invoke - keep connection pool alive throughout HITL
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                try:
                    result = loop.run_until_complete(_invoke_with_hitl())
                except Exception as e:
                    logger.error(f"Error invoking graph: {e}")
                    print(f"\n❌ Error: {e}")
                    continue

                # After all interrupts handled, display the final response
                print("\n🤖 Assistant: ", end="", flush=True)

                response_content = ""
                structured_response = None
                try:
                    # Debug: Log what's in the result
                    logger.debug(f"Result keys: {result.keys() if result else 'None'}")
                    if result:
                        for key in result.keys():
                            logger.debug(f"Result['{key}'] type: {type(result[key])}")

                    # Get the latest messages from the result
                    if result and "messages" in result:
                        latest_messages = result["messages"]
                        # Find the last AI message
                        for msg in reversed(latest_messages):
                            if isinstance(msg, AIMessage):
                                if hasattr(msg, "content") and msg.content:
                                    response_content = msg.content
                                    print(response_content, end="", flush=True)
                                    break

                    # Check for structured output and display it separately
                    if result and "structured_response" in result:
                        structured_response = result["structured_response"]
                        import json

                        structured_json = json.dumps(
                            structured_response.model_dump()
                            if hasattr(structured_response, "model_dump")
                            else structured_response,
                            indent=2,
                        )

                        # If there was message content, add separator
                        if response_content.strip():
                            print("\n\n📊 Structured Output:")
                            print(structured_json)
                        else:
                            # No message content, just show structured output
                            print(structured_json, end="", flush=True)

                        response_content = response_content or structured_json

                    print()  # New line after response

                    # Add assistant response to history if we got content
                    if response_content.strip():
                        assistant_message = AIMessage(content=response_content)
                        messages.append(assistant_message)
                    else:
                        print("(No response content generated)")

                except Exception as e:
                    print(f"\n❌ Error processing response: {e}")
                    print(f"Stack trace:\n{traceback.format_exc()}")
                    logger.error(f"Response processing error: {e}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")

            except EOFError:
                # Handle Ctrl-D
                print("\n\n👋 Goodbye! Chat session ended.")
                break
            except KeyboardInterrupt:
                # Handle Ctrl-C
                print("\n\n👋 Chat session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Chat error: {e}")
                traceback.print_exc()

    except Exception as e:
        logger.error(f"Failed to initialize chat session: {e}")
        print(f"❌ Failed to start chat session: {e}")
        sys.exit(1)


def handle_schema_command(options: Namespace) -> None:
    logger.debug("Generating JSON schema...")
    print(json.dumps(AppConfig.model_json_schema(), indent=2))


def handle_graph_command(options: Namespace) -> None:
    logger.debug("Generating graph representation...")
    config: AppConfig = AppConfig.from_file(options.config)
    app = create_dao_ai_graph(config)
    save_image(app, options.output)


def handle_deploy_command(options: Namespace) -> None:
    from dao_ai.config import DeploymentTarget

    logger.debug(f"Validating configuration from {options.config}...")
    try:
        config: AppConfig = AppConfig.from_file(options.config)

        # Hybrid target resolution:
        # 1. CLI --target takes precedence
        # 2. Fall back to config.app.deployment_target
        # 3. Default to MODEL_SERVING (handled in deploy_agent)
        target: DeploymentTarget | None = None
        if options.target is not None:
            target = DeploymentTarget(options.target)
            logger.info(f"Using CLI-specified deployment target: {target.value}")
        elif config.app is not None and config.app.deployment_target is not None:
            target = config.app.deployment_target
            logger.info(f"Using config file deployment target: {target.value}")
        else:
            logger.info("No deployment target specified, defaulting to model_serving")

        config.create_agent()
        config.deploy_agent(target=target)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


def handle_validate_command(options: Namespace) -> None:
    logger.debug(f"Validating configuration from {options.config}...")
    try:
        config: AppConfig = AppConfig.from_file(options.config)
        _ = create_dao_ai_graph(config)
        config.model_dump(by_alias=True)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)


def _format_schema_pretty(schema: dict[str, Any], indent: int = 0) -> str:
    """
    Format a JSON schema in a more readable, concise format.

    Args:
        schema: The JSON schema to format
        indent: Current indentation level

    Returns:
        Pretty-formatted schema string
    """
    if not schema:
        return ""

    lines: list[str] = []
    indent_str = "  " * indent

    # Get required fields
    required_fields = set(schema.get("required", []))

    # Handle object type with properties
    if schema.get("type") == "object" and "properties" in schema:
        properties = schema["properties"]

        for prop_name, prop_schema in properties.items():
            is_required = prop_name in required_fields
            req_marker = " (required)" if is_required else " (optional)"

            prop_type = prop_schema.get("type", "any")
            prop_desc = prop_schema.get("description", "")

            # Handle different types
            if prop_type == "array":
                items = prop_schema.get("items", {})
                item_type = items.get("type", "any")
                type_str = f"array<{item_type}>"
            elif prop_type == "object":
                type_str = "object"
            else:
                type_str = prop_type

            # Format enum values if present
            if "enum" in prop_schema:
                enum_values = ", ".join(str(v) for v in prop_schema["enum"])
                type_str = f"{type_str} (one of: {enum_values})"

            # Build the line
            line = f"{indent_str}{prop_name}: {type_str}{req_marker}"
            if prop_desc:
                line += f"\n{indent_str}  └─ {prop_desc}"

            lines.append(line)

            # Recursively handle nested objects
            if prop_type == "object" and "properties" in prop_schema:
                nested = _format_schema_pretty(prop_schema, indent + 1)
                if nested:
                    lines.append(nested)

    # Handle simple types without properties
    elif "type" in schema:
        schema_type = schema["type"]
        if schema.get("description"):
            lines.append(f"{indent_str}Type: {schema_type}")
            lines.append(f"{indent_str}└─ {schema['description']}")
        else:
            lines.append(f"{indent_str}Type: {schema_type}")

    return "\n".join(lines)


def handle_list_mcp_tools_command(options: Namespace) -> None:
    """
    List available MCP tools from configuration.

    Shows all MCP servers and their available tools, indicating which
    are included/excluded based on filter configuration.
    """
    logger.debug(f"Listing MCP tools from configuration: {options.config}")

    try:
        from dao_ai.config import McpFunctionModel
        from dao_ai.tools.mcp import MCPToolInfo, _matches_pattern, list_mcp_tools

        # Load configuration
        config: AppConfig = AppConfig.from_file(options.config)

        # Find all MCP tools in configuration
        mcp_tools_config: list[tuple[str, McpFunctionModel]] = []
        if config.tools:
            for tool_name, tool_model in config.tools.items():
                logger.debug(
                    f"Checking tool: {tool_name}, function type: {type(tool_model.function)}"
                )
                if tool_model.function and isinstance(
                    tool_model.function, McpFunctionModel
                ):
                    mcp_tools_config.append((tool_name, tool_model.function))

        if not mcp_tools_config:
            logger.warning("No MCP tools found in configuration")
            print("\n⚠️  No MCP tools configured in this file.")
            print(f"   Configuration: {options.config}")
            print(
                "\nTo add MCP tools, define them in the 'tools' section with 'type: mcp'"
            )
            sys.exit(0)

        # Collect all results first (aggregate before displaying)
        results: list[dict[str, Any]] = []
        for tool_name, mcp_function in mcp_tools_config:
            result = {
                "tool_name": tool_name,
                "mcp_function": mcp_function,
                "error": None,
                "all_tools": [],
                "included_tools": [],
                "excluded_tools": [],
            }

            try:
                logger.info(f"Connecting to MCP server: {mcp_function.mcp_url}")

                # Get all available tools (unfiltered)
                all_tools: list[MCPToolInfo] = list_mcp_tools(
                    mcp_function, apply_filters=False
                )

                # Get filtered tools (what will actually be loaded)
                filtered_tools: list[MCPToolInfo] = list_mcp_tools(
                    mcp_function, apply_filters=True
                )

                included_names = {t.name for t in filtered_tools}

                # Categorize tools
                for tool in sorted(all_tools, key=lambda t: t.name):
                    if tool.name in included_names:
                        result["included_tools"].append(tool)
                    else:
                        # Determine why it was excluded
                        reason = ""
                        if mcp_function.exclude_tools:
                            if _matches_pattern(tool.name, mcp_function.exclude_tools):
                                matching_patterns = [
                                    p
                                    for p in mcp_function.exclude_tools
                                    if _matches_pattern(tool.name, [p])
                                ]
                                reason = f" (matches exclude pattern: {', '.join(matching_patterns)})"
                        if not reason and mcp_function.include_tools:
                            reason = " (not in include list)"
                        result["excluded_tools"].append((tool, reason))

                result["all_tools"] = all_tools

            except KeyboardInterrupt:
                result["error"] = "Connection interrupted by user"
                results.append(result)
                break
            except Exception as e:
                logger.error(f"Failed to list tools from MCP server: {e}")
                result["error"] = str(e)

            results.append(result)

        # Now display all results at once (no logging interleaving)
        print(f"\n{'=' * 80}")
        print("MCP TOOLS DISCOVERY")
        print(f"Configuration: {options.config}")
        print(f"{'=' * 80}\n")

        for result in results:
            tool_name = result["tool_name"]
            mcp_function = result["mcp_function"]

            print(f"📦 Tool: {tool_name}")
            print(f"   Server: {mcp_function.mcp_url}")

            # Show connection type
            if mcp_function.connection:
                print(f"   Connection: UC Connection '{mcp_function.connection.name}'")
            else:
                print(f"   Transport: {mcp_function.transport.value}")

            # Show filters if configured
            if mcp_function.include_tools or mcp_function.exclude_tools:
                print("\n   Filters:")
                if mcp_function.include_tools:
                    print(f"     Include: {', '.join(mcp_function.include_tools)}")
                if mcp_function.exclude_tools:
                    print(f"     Exclude: {', '.join(mcp_function.exclude_tools)}")

            # Check for errors
            if result["error"]:
                print(f"\n   ❌ Error: {result['error']}")
                print("   Could not connect to MCP server")
                if result["error"] != "Connection interrupted by user":
                    print(
                        "   Tip: Verify server URL, authentication, and network connectivity"
                    )
            else:
                all_tools = result["all_tools"]
                included_tools = result["included_tools"]
                excluded_tools = result["excluded_tools"]

                # Show stats based on --apply-filters flag
                if options.apply_filters:
                    # Simplified view: only show filtered tools count
                    print(
                        f"\n   Available Tools: {len(included_tools)} (after filters)"
                    )
                else:
                    # Full view: show all, included, and excluded counts
                    print(f"\n   Available Tools: {len(all_tools)} total")
                    print(f"   ├─ ✓ Included: {len(included_tools)}")
                    print(f"   └─ ✗ Excluded: {len(excluded_tools)}")

                # Show included tools with FULL descriptions and schemas
                if included_tools:
                    if options.apply_filters:
                        print(f"\n   Tools ({len(included_tools)}):")
                    else:
                        print(f"\n   ✓ Included Tools ({len(included_tools)}):")

                    for tool in included_tools:
                        print(f"\n     • {tool.name}")
                        if tool.description:
                            # Show full description (no truncation)
                            print(f"       Description: {tool.description}")
                        if tool.input_schema:
                            # Pretty print schema in readable format
                            print("       Parameters:")
                            pretty_schema = _format_schema_pretty(
                                tool.input_schema, indent=0
                            )
                            if pretty_schema:
                                # Indent the schema for better readability
                                for line in pretty_schema.split("\n"):
                                    print(f"         {line}")
                            else:
                                print("         (none)")

                # Show excluded tools only if NOT applying filters
                if excluded_tools and not options.apply_filters:
                    print(f"\n   ✗ Excluded Tools ({len(excluded_tools)}):")
                    for tool, reason in excluded_tools:
                        print(f"     • {tool.name}{reason}")

            print(f"\n{'-' * 80}\n")

        # Summary
        print(f"{'=' * 80}")
        print(f"Summary: Found {len(mcp_tools_config)} MCP server(s)")
        print(f"{'=' * 80}\n")

        sys.exit(0)

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {options.config}")
        print(f"\n❌ Error: Configuration file not found: {options.config}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to list MCP tools: {e}")
        logger.debug(traceback.format_exc())
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def setup_logging(verbosity: int) -> None:
    levels: dict[int, str] = {
        0: "ERROR",
        1: "WARNING",
        2: "INFO",
        3: "DEBUG",
        4: "TRACE",
    }
    level: str = levels.get(verbosity, "TRACE")
    configure_logging(level=level)


def generate_bundle_from_template(config_path: Path, app_name: str) -> Path:
    """
    Generate an app-specific databricks.yaml from databricks.yaml.template.

    This function:
    1. Reads databricks.yaml.template (permanent template file)
    2. Replaces __APP_NAME__ with the actual app name
    3. Writes to databricks.yaml (overwrites if exists)
    4. Returns the path to the generated file

    The generated databricks.yaml is overwritten on each deployment and is not tracked in git.
    The template contains cloud-specific targets (azure, aws, gcp) with appropriate node types.

    Args:
        config_path: Path to the app config file
        app_name: Normalized app name

    Returns:
        Path to the generated databricks.yaml file
    """
    cwd = Path.cwd()
    template_path = cwd / "databricks.yaml.template"
    output_path = cwd / "databricks.yaml"

    if not template_path.exists():
        logger.error(f"Template file {template_path} does not exist.")
        sys.exit(1)

    # Read template
    with open(template_path, "r") as f:
        template_content = f.read()

    # Replace template variables
    bundle_content = template_content.replace("__APP_NAME__", app_name)

    # Write generated databricks.yaml (overwrite if exists)
    with open(output_path, "w") as f:
        f.write(bundle_content)

    logger.info(f"Generated bundle configuration at {output_path} from template")
    return output_path


def run_databricks_command(
    command: list[str],
    profile: Optional[str] = None,
    config: Optional[str] = None,
    target: Optional[str] = None,
    cloud: Optional[str] = None,
    dry_run: bool = False,
    deployment_target: Optional[str] = None,
) -> None:
    """Execute a databricks CLI command with optional profile, target, and cloud.

    Args:
        command: The databricks CLI command to execute (e.g., ["bundle", "deploy"])
        profile: Optional Databricks CLI profile name
        config: Optional path to the configuration file
        target: Optional bundle target name (if not provided, auto-generated from app name and cloud)
        cloud: Optional cloud provider ('azure', 'aws', 'gcp'). Auto-detected if not specified.
        dry_run: If True, print the command without executing
        deployment_target: Optional agent deployment target ('model_serving' or 'apps').
            Passed to the deploy notebook via bundle variable.
    """
    config_path = Path(config) if config else None

    if config_path and not config_path.exists():
        logger.error(f"Configuration file {config_path} does not exist.")
        sys.exit(1)

    # Load app config
    app_config: AppConfig = AppConfig.from_file(config_path) if config_path else None
    normalized_name: str = normalize_name(app_config.app.name) if app_config else None

    # Auto-detect cloud provider if not specified (used for node_type selection)
    if not cloud:
        cloud = detect_cloud_provider(profile)
        if cloud:
            logger.info(f"Auto-detected cloud provider: {cloud}")
        else:
            logger.warning("Could not detect cloud provider. Defaulting to 'azure'.")
            cloud = "azure"

    # Generate app-specific bundle from template (overwrites databricks.yaml temporarily)
    if config_path and app_config:
        generate_bundle_from_template(config_path, normalized_name)

    # Use app-specific cloud target: {app_name}-{cloud}
    # This ensures each app has unique deployment identity while supporting cloud-specific settings
    # Can be overridden with explicit --target
    if not target:
        target = f"{normalized_name}-{cloud}"
        logger.info(f"Using app-specific cloud target: {target}")

    # Build databricks command
    # --profile is a global flag, --target is a subcommand flag for 'bundle'
    cmd = ["databricks"]
    if profile:
        cmd.extend(["--profile", profile])

    cmd.extend(command)

    # --target must come after the bundle subcommand (it's a subcommand-specific flag)
    if target:
        cmd.extend(["--target", target])

    # Add config_path variable for notebooks
    if config_path and app_config:
        # Calculate relative path from notebooks directory to config file
        config_abs = config_path.resolve()
        cwd = Path.cwd()
        notebooks_dir = cwd / "notebooks"

        try:
            relative_config = config_abs.relative_to(notebooks_dir)
        except ValueError:
            relative_config = Path(os.path.relpath(config_abs, notebooks_dir))

        cmd.append(f'--var="config_path={relative_config}"')

    # Add deployment_target variable for notebooks (hybrid resolution)
    # Priority: CLI arg > config file > default (model_serving)
    resolved_deployment_target: str = "model_serving"
    if deployment_target is not None:
        resolved_deployment_target = deployment_target
        logger.debug(
            f"Using CLI-specified deployment target: {resolved_deployment_target}"
        )
    elif app_config and app_config.app and app_config.app.deployment_target:
        # deployment_target is DeploymentTarget enum (str subclass) or string
        # str() works for both since DeploymentTarget inherits from str
        resolved_deployment_target = str(app_config.app.deployment_target)
        logger.debug(
            f"Using config file deployment target: {resolved_deployment_target}"
        )
    else:
        logger.debug("Using default deployment target: model_serving")

    cmd.append(f'--var="deployment_target={resolved_deployment_target}"')

    logger.debug(f"Executing command: {' '.join(cmd)}")

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        for line in iter(process.stdout.readline, ""):
            print(line.rstrip())

        process.wait()

        if process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}")
            sys.exit(1)
        else:
            logger.info("Command executed successfully")

    except FileNotFoundError:
        logger.error("databricks CLI not found. Please install the Databricks CLI.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        sys.exit(1)


def handle_bundle_command(options: Namespace) -> None:
    logger.debug("Bundling configuration...")
    profile: Optional[str] = options.profile
    config: Optional[str] = options.config
    target: Optional[str] = options.target
    cloud: Optional[str] = options.cloud
    dry_run: bool = options.dry_run
    deployment_target: Optional[str] = options.deployment_target

    if options.deploy:
        logger.info("Deploying DAO AI asset bundle...")
        run_databricks_command(
            ["bundle", "deploy"],
            profile=profile,
            config=config,
            target=target,
            cloud=cloud,
            dry_run=dry_run,
            deployment_target=deployment_target,
        )
    if options.run:
        logger.info("Running DAO AI system with current configuration...")
        # Use static job resource key that matches databricks.yaml (resources.jobs.deploy_job)
        run_databricks_command(
            ["bundle", "run", "deploy_job"],
            profile=profile,
            config=config,
            target=target,
            cloud=cloud,
            dry_run=dry_run,
            deployment_target=deployment_target,
        )
    if options.destroy:
        logger.info("Destroying DAO AI system with current configuration...")
        run_databricks_command(
            ["bundle", "destroy", "--auto-approve"],
            profile=profile,
            config=config,
            target=target,
            cloud=cloud,
            dry_run=dry_run,
            deployment_target=deployment_target,
        )
    else:
        logger.warning("No action specified. Use --deploy, --run or --destroy flags.")


def main() -> None:
    options: argparse.Namespace = parse_args(sys.argv[1:])
    setup_logging(options.verbose)
    match options.command:
        case "schema":
            handle_schema_command(options)
        case "validate":
            handle_validate_command(options)
        case "graph":
            handle_graph_command(options)
        case "bundle":
            handle_bundle_command(options)
        case "deploy":
            handle_deploy_command(options)
        case "chat":
            handle_chat_command(options)
        case "list-mcp-tools":
            handle_list_mcp_tools_command(options)
        case _:
            logger.error(f"Unknown command: {options.command}")
            sys.exit(1)


if __name__ == "__main__":
    main()
