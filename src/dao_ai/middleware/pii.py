"""
PII detection middleware for DAO AI agents.

Detects and handles Personally Identifiable Information (PII) in conversations
using configurable strategies (redact, mask, hash, block).

Example:
    from dao_ai.middleware import create_pii_middleware

    # Redact emails in user input
    middleware = create_pii_middleware(
        pii_type="email",
        strategy="redact",
        apply_to_input=True,
    )
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Pattern

from langchain.agents.middleware import PIIMiddleware
from loguru import logger

__all__ = [
    "PIIMiddleware",
    "create_pii_middleware",
]

# Type alias for PII detector
PIIDetector = str | Pattern[str] | Callable[[str], list[dict[str, str | int]]]

# Built-in PII types
BUILTIN_PII_TYPES = frozenset({"email", "credit_card", "ip", "mac_address", "url"})


def create_pii_middleware(
    pii_type: str,
    strategy: Literal["redact", "mask", "hash", "block"] = "redact",
    detector: PIIDetector | None = None,
    apply_to_input: bool = True,
    apply_to_output: bool = False,
    apply_to_tool_results: bool = False,
) -> PIIMiddleware:
    """
    Create a PIIMiddleware for detecting and handling PII.

    Detects Personally Identifiable Information in conversations and handles
    it according to the specified strategy. Useful for compliance, privacy,
    and sanitizing logs.

    Built-in PII types:
    - email: Email addresses
    - credit_card: Credit card numbers (Luhn validated)
    - ip: IP addresses
    - mac_address: MAC addresses
    - url: URLs

    Args:
        pii_type: Type of PII to detect. Use built-in types (email, credit_card,
            ip, mac_address, url) or custom type names with a detector.
        strategy: How to handle detected PII:
            - "redact": Replace with [REDACTED_{TYPE}] (default)
            - "mask": Partially obscure (e.g., ****-****-****-1234)
            - "hash": Replace with deterministic hash
            - "block": Raise exception when detected
        detector: Custom detector for non-built-in types. Can be:
            - str: Regex pattern string
            - re.Pattern: Compiled regex pattern
            - Callable: Function(content: str) -> list[dict] with keys:
                - text: The matched text
                - start: Start index
                - end: End index
            Default None (uses built-in detector for built-in types).
        apply_to_input: Check user messages before model call. Default True.
        apply_to_output: Check AI messages after model call. Default False.
        apply_to_tool_results: Check tool results after execution. Default False.

    Returns:
        List containing PIIMiddleware instance

    Raises:
        ValueError: If custom pii_type without detector, or invalid strategy

    Example:
        # Redact emails in input
        email_redactor = create_pii_middleware(
            pii_type="email",
            strategy="redact",
            apply_to_input=True,
        )

        # Mask credit cards
        card_masker = create_pii_middleware(
            pii_type="credit_card",
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,
        )

        # Block API keys with custom regex
        api_key_blocker = create_pii_middleware(
            pii_type="api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
        )

        # Custom SSN detector with validation
        def detect_ssn(content: str) -> list[dict]:
            matches = []
            pattern = r"\\d{3}-\\d{2}-\\d{4}"
            for match in re.finditer(pattern, content):
                ssn = match.group(0)
                first_three = int(ssn[:3])
                if first_three not in [0, 666] and not (900 <= first_three <= 999):
                    matches.append({
                        "text": ssn,
                        "start": match.start(),
                        "end": match.end(),
                    })
            return matches

        ssn_hasher = create_pii_middleware(
            pii_type="ssn",
            detector=detect_ssn,
            strategy="hash",
        )
    """
    # Validate: custom types require detector
    if pii_type not in BUILTIN_PII_TYPES and detector is None:
        raise ValueError(
            f"Custom PII type '{pii_type}' requires a detector. "
            f"Built-in types are: {', '.join(sorted(BUILTIN_PII_TYPES))}"
        )

    logger.debug(
        "Creating PII middleware",
        pii_type=pii_type,
        strategy=strategy,
        has_custom_detector=detector is not None,
        apply_to_input=apply_to_input,
        apply_to_output=apply_to_output,
        apply_to_tool_results=apply_to_tool_results,
    )

    # Build kwargs
    kwargs: dict[str, Any] = {
        "strategy": strategy,
        "apply_to_input": apply_to_input,
        "apply_to_output": apply_to_output,
        "apply_to_tool_results": apply_to_tool_results,
    }

    if detector is not None:
        kwargs["detector"] = detector

    return PIIMiddleware(pii_type, **kwargs)
