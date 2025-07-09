import argparse
import json
import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Sequence

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from dao_ai.config import AppConfig
from dao_ai.graph import create_dao_ai_graph
from dao_ai.models import save_image
from dao_ai.utils import normalize_name

logger.remove()
logger.add(sys.stderr, level="ERROR")


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
    )
    bundle_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without executing the deployment or run commands",
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
  dao-ai chat -c config/model_config.yaml                              # Start chat with default settings
  dao-ai chat -c config/retail.yaml --custom-input store_num=87887     # Chat with custom store number
  dao-ai chat -c config/prod.yaml --user-id john123                    # Chat with specific user ID
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
        default="my_user_id",
        metavar="ID",
        help="User ID for the chat session (default: my_user_id)",
    )
    chat_parser.add_argument(
        "--thread-id",
        type=str,
        default="1",
        metavar="ID",
        help="Thread ID for the chat session (default: 1)",
    )

    options = parser.parse_args(args)

    return options


def handle_chat_command(options: Namespace) -> None:
    """Interactive chat REPL with the DAO AI system."""
    logger.debug("Starting chat session with DAO AI system...")

    try:
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
        from langchain_core.messages import HumanMessage

        from dao_ai.models import process_messages_stream

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

                # Prepare custom inputs for the agent
                custom_inputs = {"configurable": configurable}

                print("\n🤖 Assistant: ", end="", flush=True)

                # Stream the response
                response_content = ""
                try:
                    for chunk in process_messages_stream(app, messages, custom_inputs):
                        # Handle different chunk types
                        if hasattr(chunk, "content") and chunk.content:
                            content = chunk.content
                            print(content, end="", flush=True)
                            response_content += content
                        elif hasattr(chunk, "choices") and chunk.choices:
                            # Handle ChatCompletionChunk format
                            for choice in chunk.choices:
                                if (
                                    hasattr(choice, "delta")
                                    and choice.delta
                                    and choice.delta.content
                                ):
                                    content = choice.delta.content
                                    print(content, end="", flush=True)
                                    response_content += content

                    print()  # New line after streaming

                    # Add assistant response to history if we got content
                    if response_content.strip():
                        from langchain_core.messages import AIMessage

                        assistant_message = AIMessage(content=response_content)
                        messages.append(assistant_message)
                    else:
                        print("(No response content generated)")

                except Exception as e:
                    print(f"\n❌ Error during streaming: {e}")
                    logger.error(f"Streaming error: {e}")

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
    logger.debug(f"Validating configuration from {options.config}...")
    try:
        config: AppConfig = AppConfig.from_file(options.config)
        config.create_agent()
        config.deploy_agent()
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


def setup_logging(verbosity: int) -> None:
    logger.remove()
    levels: dict[int, str] = {
        0: "ERROR",
        1: "WARNING",
        2: "INFO",
        3: "DEBUG",
        4: "TRACE",
    }
    level: str = levels.get(verbosity, "TRACE")
    logger.add(sys.stderr, level=level)


def run_databricks_command(
    command: list[str],
    profile: Optional[str] = None,
    config: Optional[str] = None,
    target: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Execute a databricks CLI command with optional profile."""
    cmd = ["databricks"]
    if profile:
        cmd.extend(["--profile", profile])
    if target:
        cmd.extend(["--target", target])
    cmd.extend(command)
    if config:
        config_path = Path(config)

        if not config_path.exists():
            logger.error(f"Configuration file {config_path} does not exist.")
            sys.exit(1)

        app_config: AppConfig = AppConfig.from_file(config_path)

        # Always convert to path relative to notebooks directory
        # Get absolute path of config file and current working directory
        config_abs = config_path.resolve()
        cwd = Path.cwd()
        notebooks_dir = cwd / "notebooks"

        # Calculate relative path from notebooks directory to config file
        try:
            relative_config = config_abs.relative_to(notebooks_dir)
        except ValueError:
            # Config file is outside notebooks directory, calculate relative path
            # Use os.path.relpath to get the relative path from notebooks_dir to config file
            relative_config = Path(os.path.relpath(config_abs, notebooks_dir))

        cmd.append(f'--var="config_path={relative_config}"')

        normalized_name: str = normalize_name(app_config.app.name)
        cmd.append(f'--var="app_name={normalized_name}"')

    logger.debug(f"Executing command: {' '.join(cmd)}")

    if dry_run:
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


def handle_bundle_command(options: Namespace) -> None:
    logger.debug("Bundling configuration...")
    profile: Optional[str] = options.profile
    config: Optional[str] = options.config
    target: Optional[str] = options.target
    dry_run: bool = options.dry_run
    if options.deploy:
        logger.info("Deploying DAO AI asset bundle...")
        run_databricks_command(
            ["bundle", "deploy"], profile, config, target, dry_run=dry_run
        )
    if options.run:
        logger.info("Running DAO AI system with current configuration...")
        run_databricks_command(
            ["bundle", "run", "deploy-end-to-end"],
            profile,
            config,
            target,
            dry_run=dry_run,
        )
    if options.destroy:
        logger.info("Destroying DAO AI system with current configuration...")
        run_databricks_command(
            ["bundle", "destroy", "--auto-approve"],
            profile,
            config,
            target,
            dry_run=dry_run,
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
        case _:
            logger.error(f"Unknown command: {options.command}")
            sys.exit(1)


if __name__ == "__main__":
    main()
