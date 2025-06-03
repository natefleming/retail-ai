import argparse
import json
import sys
from argparse import ArgumentParser, Namespace
from typing import Sequence

from loguru import logger
from mlflow.models import ModelConfig

from retail_ai.config import AppConfig

logger.remove()
logger.add(sys.stderr, level="ERROR")


def parse_args(args: Sequence[str]) -> Namespace:
    parser: ArgumentParser = ArgumentParser(description="The Retail AI CLI")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase the verbosity"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    _: ArgumentParser = subparsers.add_parser("schema", help="Generate the json schema")

    validation_parser: ArgumentParser = subparsers.add_parser(
        "validate", help="Validate the configuration"
    )
    validation_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="model_config.yaml",
        help="Path to the model configuration file (default: model_config.yaml)",
    )

    options = parser.parse_args(args)

    return options


def handle_schema_command(options: Namespace) -> None:
    logger.debug("Generating JSON schema...")
    print(json.dumps(AppConfig.model_json_schema(), indent=2))


def handle_validate_command(options: Namespace) -> None:
    logger.debug(f"Validating configuration from {options.config}...")
    model_config: ModelConfig = ModelConfig(development_config=options.config)
    _: AppConfig = AppConfig(**model_config.to_dict())


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
        case _:
            logger.error(f"Unknown command: {options.command}")
            sys.exit(1)


if __name__ == "__main__":
    main()
