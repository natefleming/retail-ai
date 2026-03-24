"""Auto-install guardrails-ai hub validators declared in config.

This module provides ``ensure_guardrails_hub`` which is called during
``AppConfig.initialize()``.  It collects all ``hub`` URIs from
guardrail configurations and, when the ``GUARDRAILSAI_API_KEY``
environment variable is present (or a pre-existing token exists in
``~/.guardrailsrc``), automatically installs any validators that are
not yet available in ``guardrails.hub``.

If no guardrail in the config declares a ``hub`` field, the function
is a complete no-op and never imports or interacts with ``guardrails``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from dao_ai.config import AppConfig

GUARDRAILSAI_API_KEY_ENV = "GUARDRAILSAI_API_KEY"


def _collect_hub_uris(config: AppConfig) -> set[str]:
    """Gather all unique ``hub`` URIs from guardrail configs.

    Checks both the top-level ``guardrails`` section and the
    ``guardrails`` list on each agent.
    """
    uris: set[str] = set()

    for guardrail in config.guardrails.values():
        if guardrail.hub:
            uris.add(guardrail.hub)

    for agent in config.agents.values():
        for guardrail in agent.guardrails:
            if guardrail.hub:
                uris.add(guardrail.hub)

    return uris


def _hub_uri_to_registry_key(hub_uri: str) -> str:
    """Extract the registry key from a hub URI.

    ``hub://guardrails/toxic_language`` -> ``guardrails/toxic_language``
    """
    prefix = "hub://"
    if hub_uri.startswith(prefix):
        return hub_uri[len(prefix) :].rstrip("/")
    return hub_uri.rstrip("/")


def _is_validator_installed(hub_uri: str) -> bool:
    """Return True if the validator module is actually importable.

    The guardrails registry (``~/.guardrailsrc``) can contain stale
    entries when the virtualenv is recreated, so we verify that the
    underlying Python package is present rather than just checking the
    registry.  This mirrors the import that ``guardrails.hub.__getattr__``
    performs when MLflow resolves a validator class.
    """
    registry_key: str = _hub_uri_to_registry_key(hub_uri)
    slug: str = registry_key.split("/", 1)[-1]
    module_name: str = f"guardrails_grhub_{slug}"
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        logger.debug(
            "Validator module not found despite possible registry entry",
            hub_uri=hub_uri,
            module=module_name,
        )
        return False


def _resolve_secret_template(template: str) -> str | None:
    """Resolve a ``{{secrets/scope/key}}`` template via Databricks Secrets API.

    Returns the secret value on Databricks clusters, or ``None`` if the
    secret cannot be resolved (e.g. running locally without auth).
    """
    inner = template[len("{{secrets/") : -len("}}")]
    parts = inner.split("/", 1)
    if len(parts) != 2:
        logger.debug("Malformed secret template", template=template)
        return None

    scope, key = parts
    try:
        from dao_ai.providers.databricks import DatabricksProvider

        provider = DatabricksProvider()
        value: str | None = provider.get_secret(scope, key)
        if value:
            logger.debug(
                "Resolved API key from Databricks secret",
                scope=scope,
                key=key,
            )
        return value
    except Exception as e:
        logger.debug(
            "Could not resolve secret template -- "
            "this is expected when running outside Databricks",
            template=template,
            error=str(e),
        )
        return None


def _configure_token(api_key: str) -> None:
    """Set the guardrails-ai token in memory with recommended settings.

    Equivalent to:
        guardrails configure --token $KEY --disable-metrics --enable-remote-inferencing
    """
    from guardrails.classes.rc import RC
    from guardrails.settings import Settings

    logger.info(
        "Configuring guardrails-ai from environment variable",
        env_var=GUARDRAILSAI_API_KEY_ENV,
        enable_metrics=False,
        use_remote_inferencing=True,
    )
    settings = Settings()
    settings.rc = RC(
        token=api_key,
        enable_metrics=False,
        use_remote_inferencing=True,
    )
    logger.debug("Guardrails-ai token configured successfully")


def _has_existing_token() -> bool:
    """Check whether a valid token already exists in ``~/.guardrailsrc``."""
    try:
        from guardrails.settings import Settings

        settings = Settings()
        has_token: bool = bool(settings.rc and settings.rc.token)
        if has_token:
            logger.debug("Found existing guardrails-ai token in ~/.guardrailsrc")
        return has_token
    except Exception as e:
        logger.debug(
            "Could not read existing guardrails-ai configuration",
            error=str(e),
        )
        return False


def _ensure_token_configured() -> bool:
    """Ensure a guardrails-ai token is available.

    Checks, in order:
    1. ``GUARDRAILSAI_API_KEY`` environment variable
    2. Existing token in ``~/.guardrailsrc``

    Returns ``True`` if a token is available, ``False`` otherwise.
    """
    api_key: str | None = os.environ.get(GUARDRAILSAI_API_KEY_ENV)

    if api_key:
        logger.debug(
            "Found guardrails-ai authentication key",
            env_var=GUARDRAILSAI_API_KEY_ENV,
        )
        _configure_token(api_key)
        return True

    if _has_existing_token():
        logger.info(
            "Using existing guardrails-ai token from ~/.guardrailsrc "
            f"({GUARDRAILSAI_API_KEY_ENV} env var not set)"
        )
        return True

    return False


def ensure_single_hub_validator(hub_uri: str) -> bool:
    """Ensure a single guardrails-ai hub validator is installed.

    Handles the full lifecycle for one hub URI: verifying the
    ``guardrails`` package is importable, configuring authentication,
    and installing the validator if it is not already present.

    Args:
        hub_uri: Hub URI (e.g. ``"hub://guardrails/toxic_language"``).

    Returns:
        ``True`` if the validator is available after the call (either
        already installed or freshly installed), ``False`` on failure.
    """
    try:
        import guardrails  # noqa: F811
    except ImportError:
        logger.error(
            f"Hub URI '{hub_uri}' requires the 'guardrails-ai' package. "
            "Install it with: pip install 'guardrails-ai>=0.9.2'"
        )
        return False

    registry_key: str = _hub_uri_to_registry_key(hub_uri)

    if _is_validator_installed(hub_uri):
        logger.debug(
            "Hub validator already installed -- skipping",
            registry_key=registry_key,
            hub_uri=hub_uri,
        )
        return True

    if not _ensure_token_configured():
        logger.warning(
            f"Cannot install hub validator '{registry_key}' -- no API key available. "
            f"Set the {GUARDRAILSAI_API_KEY_ENV} environment variable or run "
            "'guardrails configure --token <TOKEN>'."
        )
        return False

    logger.info(
        "Installing guardrails hub validator",
        registry_key=registry_key,
        hub_uri=hub_uri,
    )
    try:
        guardrails.install(hub_uri, quiet=True)
        logger.info(
            "Successfully installed hub validator",
            registry_key=registry_key,
        )
        return True
    except Exception as e:
        logger.error(
            "Failed to install hub validator",
            registry_key=registry_key,
            hub_uri=hub_uri,
            error=str(e),
        )
        return False


def ensure_guardrails_hub(config: AppConfig) -> None:
    """Auto-configure guardrails-ai and install hub validators if needed.

    No-op when no guardrails in the config have a ``hub`` attribute.
    When hub URIs are present, requires ``GUARDRAILSAI_API_KEY`` env var
    (or a pre-existing ``~/.guardrailsrc`` token).  Skips validators
    that are already installed.
    """
    hub_uris: set[str] = _collect_hub_uris(config)

    if not hub_uris:
        logger.debug("No guardrails hub URIs found in config -- skipping hub setup")
        return

    try:
        import guardrails  # noqa: F811
    except ImportError:
        logger.error(
            "Guardrails hub URIs declared in config but the 'guardrails-ai' "
            "package is not installed. Install it with: "
            "pip install 'guardrails-ai>=0.9.2'"
        )
        return

    logger.info(
        "Guardrails hub URIs found in config",
        count=len(hub_uris),
        hub_uris=sorted(hub_uris),
    )

    api_key: str | None = os.environ.get(GUARDRAILSAI_API_KEY_ENV)

    if not api_key and config.app and config.app.environment_vars:
        config_value = config.app.environment_vars.get(GUARDRAILSAI_API_KEY_ENV)
        if config_value and isinstance(config_value, str):
            if config_value.startswith("{{secrets/") and config_value.endswith("}}"):
                api_key = _resolve_secret_template(config_value)
            else:
                api_key = config_value
            if api_key:
                logger.debug(
                    "Resolved API key from config app.environment_vars",
                    env_var=GUARDRAILSAI_API_KEY_ENV,
                )

    if api_key:
        logger.debug(
            "Found guardrails-ai authentication key",
            env_var=GUARDRAILSAI_API_KEY_ENV,
        )
        _configure_token(api_key)
    elif _has_existing_token():
        logger.info(
            "Using existing guardrails-ai token from ~/.guardrailsrc "
            f"({GUARDRAILSAI_API_KEY_ENV} env var not set)"
        )
    else:
        logger.warning(
            "Guardrails hub URIs declared in config but no API key available. "
            f"Set the {GUARDRAILSAI_API_KEY_ENV} environment variable or run "
            "'guardrails configure --token <TOKEN>' to enable auto-install."
        )
        return

    installed_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0

    for hub_uri in sorted(hub_uris):
        registry_key: str = _hub_uri_to_registry_key(hub_uri)

        if _is_validator_installed(hub_uri):
            logger.debug(
                "Hub validator already installed -- skipping",
                registry_key=registry_key,
                hub_uri=hub_uri,
            )
            skipped_count += 1
            continue

        logger.info(
            "Installing guardrails hub validator",
            registry_key=registry_key,
            hub_uri=hub_uri,
        )
        try:
            guardrails.install(hub_uri, quiet=True)
            installed_count += 1
            logger.info(
                "Successfully installed hub validator",
                registry_key=registry_key,
            )
        except Exception as e:
            failed_count += 1
            logger.error(
                "Failed to install hub validator",
                registry_key=registry_key,
                hub_uri=hub_uri,
                error=str(e),
            )

    logger.info(
        "Guardrails hub setup complete",
        total=len(hub_uris),
        installed=installed_count,
        already_present=skipped_count,
        failed=failed_count,
    )


def collect_hub_code_paths(config: AppConfig) -> list[str]:
    """Return filesystem paths for installed guardrails hub validator packages.

    These paths should be included as ``code_paths`` when logging a model
    so that hub validators are available in Model Serving without runtime
    installation (which is unreliable due to network/permission constraints).

    No-op when no guardrails in the config declare a ``hub`` field.
    """
    hub_uris: set[str] = _collect_hub_uris(config)
    if not hub_uris:
        return []

    paths: list[str] = []
    for hub_uri in sorted(hub_uris):
        registry_key: str = _hub_uri_to_registry_key(hub_uri)
        slug: str = registry_key.split("/", 1)[-1]
        module_name: str = f"guardrails_grhub_{slug}"

        spec = importlib.util.find_spec(module_name)
        if spec and spec.submodule_search_locations:
            package_dir: str = spec.submodule_search_locations[0]
            paths.append(package_dir)
            logger.debug(
                "Found hub validator package for bundling",
                module=module_name,
                path=package_dir,
            )
        else:
            logger.warning(
                "Hub validator package not found locally -- "
                "install it before deploying: "
                f"guardrails hub install {hub_uri}",
                hub_uri=hub_uri,
                module=module_name,
            )

    if paths:
        logger.info(
            "Collected hub validator packages for model bundling",
            count=len(paths),
        )
    return paths
