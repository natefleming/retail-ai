"""
Tests for PII middleware factory.
"""

import re

import pytest
from langchain.agents.middleware import PIIMiddleware

from dao_ai.middleware import create_pii_middleware


class TestCreatePIIMiddleware:
    """Tests for the create_pii_middleware factory function."""

    def test_create_with_email_type(self):
        """Test creating middleware for email PII type."""
        middlewares = create_pii_middleware(pii_type="email")

        assert isinstance(middlewares, list)
        assert len(middlewares) == 1
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_credit_card_type(self):
        """Test creating middleware for credit_card PII type."""
        middlewares = create_pii_middleware(pii_type="credit_card")

        assert isinstance(middlewares, list)
        assert len(middlewares) == 1
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_ip_type(self):
        """Test creating middleware for ip PII type."""
        middlewares = create_pii_middleware(pii_type="ip")

        assert isinstance(middlewares, list)
        assert len(middlewares) == 1
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_mac_address_type(self):
        """Test creating middleware for mac_address PII type."""
        middlewares = create_pii_middleware(pii_type="mac_address")

        assert isinstance(middlewares, list)
        assert len(middlewares) == 1
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_url_type(self):
        """Test creating middleware for url PII type."""
        middlewares = create_pii_middleware(pii_type="url")

        assert isinstance(middlewares, list)
        assert len(middlewares) == 1
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_redact_strategy(self):
        """Test creating middleware with redact strategy."""
        middlewares = create_pii_middleware(
            pii_type="email",
            strategy="redact",
        )

        assert isinstance(middlewares, list)
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_mask_strategy(self):
        """Test creating middleware with mask strategy."""
        middlewares = create_pii_middleware(
            pii_type="credit_card",
            strategy="mask",
        )

        assert isinstance(middlewares, list)
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_hash_strategy(self):
        """Test creating middleware with hash strategy."""
        middlewares = create_pii_middleware(
            pii_type="email",
            strategy="hash",
        )

        assert isinstance(middlewares, list)
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_block_strategy(self):
        """Test creating middleware with block strategy."""
        middlewares = create_pii_middleware(
            pii_type="email",
            strategy="block",
        )

        assert isinstance(middlewares, list)
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_apply_to_input(self):
        """Test creating middleware with apply_to_input."""
        middlewares = create_pii_middleware(
            pii_type="email",
            apply_to_input=True,
        )

        assert isinstance(middlewares, list)
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_apply_to_output(self):
        """Test creating middleware with apply_to_output."""
        middlewares = create_pii_middleware(
            pii_type="email",
            apply_to_output=True,
        )

        assert isinstance(middlewares, list)
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_apply_to_tool_results(self):
        """Test creating middleware with apply_to_tool_results."""
        middlewares = create_pii_middleware(
            pii_type="email",
            apply_to_tool_results=True,
        )

        assert isinstance(middlewares, list)
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_all_apply_options(self):
        """Test creating middleware with all apply options."""
        middlewares = create_pii_middleware(
            pii_type="email",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        )

        assert isinstance(middlewares, list)
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_regex_string_detector(self):
        """Test creating middleware with regex string detector."""
        middlewares = create_pii_middleware(
            pii_type="api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
        )

        assert isinstance(middlewares, list)
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_compiled_regex_detector(self):
        """Test creating middleware with compiled regex detector."""
        pattern = re.compile(r"\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}")
        middlewares = create_pii_middleware(
            pii_type="phone_number",
            detector=pattern,
            strategy="mask",
        )

        assert isinstance(middlewares, list)
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_with_callable_detector(self):
        """Test creating middleware with callable detector."""

        def detect_custom(content: str) -> list[dict]:
            return [{"text": "test", "start": 0, "end": 4}]

        middlewares = create_pii_middleware(
            pii_type="custom_type",
            detector=detect_custom,
            strategy="redact",
        )

        assert isinstance(middlewares, list)
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_create_custom_type_without_detector_raises(self):
        """Test that custom type without detector raises error."""
        with pytest.raises(ValueError, match="requires a detector"):
            create_pii_middleware(pii_type="custom_type")

    def test_create_with_all_parameters(self):
        """Test creating middleware with all parameters."""
        middlewares = create_pii_middleware(
            pii_type="email",
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        )

        assert isinstance(middlewares, list)
        assert isinstance(middlewares[0], PIIMiddleware)

    def test_returns_list_for_composition(self):
        """Test that factory returns list for easy composition."""
        email_middlewares = create_pii_middleware(pii_type="email")
        card_middlewares = create_pii_middleware(pii_type="credit_card")

        assert isinstance(email_middlewares, list)
        assert isinstance(card_middlewares, list)

        # Should be composable with +
        all_middlewares = email_middlewares + card_middlewares
        assert len(all_middlewares) == 2
        assert all(isinstance(m, PIIMiddleware) for m in all_middlewares)
