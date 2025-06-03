import argparse
import json
import sys
from argparse import ArgumentParser, Namespace
from typing import Sequence

from loguru import logger
from mlflow.models import ModelConfig

from retail_ai.config import AppConfig
from retail_ai.graph import create_retail_ai_graph
from retail_ai.models import save_image

logger.remove()
logger.add(sys.stderr, level="ERROR")


def parse_args(args: Sequence[str]) -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog="retail-ai",
        description="Retail AI Agent Command Line Interface - A comprehensive tool for managing, validating, and visualizing multi-agent retail AI systems",
        epilog="""
Examples:
  retail-ai schema                                          # Generate JSON schema for configuration validation
  retail-ai validate -c config/model_config.yaml            # Validate a specific configuration file
  retail-ai graph -o architecture.png -c my_config.yaml -v  # Generate visual graph with verbose output
  retail-ai validate                                        # Validate with detailed logging
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
        help="Available commands for managing the Retail AI system",
        metavar="COMMAND",
    )

    # Schema command
    _: ArgumentParser = subparsers.add_parser(
        "schema",
        help="Generate JSON schema for configuration validation",
        description="""
Generate the JSON schema definition for the Retail AI configuration format.
This schema can be used for IDE autocompletion, validation tools, and documentation.
The output is a complete JSON Schema that describes all valid configuration options,
including agents, tools, models, orchestration patterns, and guardrails.
        """,
        epilog="Example: retail-ai schema > config_schema.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Validation command
    validation_parser: ArgumentParser = subparsers.add_parser(
        "validate",
        help="Validate configuration file syntax and semantics",
        description="""
Validate a Retail AI configuration file for correctness and completeness.
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
  retail-ai validate                                  # Validate default model_config.yaml
  retail-ai validate -c config/production.yaml       # Validate specific config file
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validation_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="model_config.yaml",
        metavar="FILE",
        help="Path to the model configuration file to validate (default: model_config.yaml)",
    )

    # Graph command
    graph_parser: ArgumentParser = subparsers.add_parser(
        "graph",
        help="Generate visual representation of the agent workflow",
        description="""
Generate a visual graph representation of the configured Retail AI system.
This creates a diagram showing:
- Agent nodes and their relationships
- Orchestration flow (supervisor or swarm patterns)
- Tool dependencies and connections
- Message routing and state transitions
- Conditional logic and decision points
        """,
        epilog="""
Examples:
  retail-ai graph -o architecture.png                # Generate PNG diagram
  retail-ai graph -o workflow.png -c prod.yaml       # Generate PNG from specific config
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
        default="model_config.yaml",
        metavar="FILE",
        help="Path to the model configuration file to visualize (default: model_config.yaml)",
    )

    options = parser.parse_args(args)

    return options


def handle_schema_command(options: Namespace) -> None:
    logger.debug("Generating JSON schema...")
    print(json.dumps(AppConfig.model_json_schema(), indent=2))


def handle_graph_command(options: Namespace) -> None:
    logger.debug("Generating graph representation...")
    model_config: ModelConfig = ModelConfig(development_config=options.config)
    config: AppConfig = AppConfig(**model_config.to_dict())
    app = create_retail_ai_graph(config)
    save_image(app, options.output)


def handle_validate_command(options: Namespace) -> None:
    logger.debug(f"Validating configuration from {options.config}...")
    try:
        model_config: ModelConfig = ModelConfig(development_config=options.config)
        _: AppConfig = AppConfig(**model_config.to_dict())
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
        case _:
            logger.error(f"Unknown command: {options.command}")
            sys.exit(1)


if __name__ == "__main__":
    main()
