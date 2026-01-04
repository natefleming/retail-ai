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
        middleware = create_pii_middleware(pii_type="email")

        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_credit_card_type(self):
        """Test creating middleware for credit_card PII type."""
        middleware = create_pii_middleware(pii_type="credit_card")

        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_ip_type(self):
        """Test creating middleware for ip PII type."""
        middleware = create_pii_middleware(pii_type="ip")

        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_mac_address_type(self):
        """Test creating middleware for mac_address PII type."""
        middleware = create_pii_middleware(pii_type="mac_address")

        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_url_type(self):
        """Test creating middleware for url PII type."""
        middleware = create_pii_middleware(pii_type="url")

        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_redact_strategy(self):
        """Test creating middleware with redact strategy."""
        middleware = create_pii_middleware(
            pii_type="email",
            strategy="redact",
        )

        # Middleware is single instance
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_mask_strategy(self):
        """Test creating middleware with mask strategy."""
        middleware = create_pii_middleware(
            pii_type="credit_card",
            strategy="mask",
        )

        # Middleware is single instance
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_hash_strategy(self):
        """Test creating middleware with hash strategy."""
        middleware = create_pii_middleware(
            pii_type="email",
            strategy="hash",
        )

        # Middleware is single instance
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_block_strategy(self):
        """Test creating middleware with block strategy."""
        middleware = create_pii_middleware(
            pii_type="email",
            strategy="block",
        )

        # Middleware is single instance
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_apply_to_input(self):
        """Test creating middleware with apply_to_input."""
        middleware = create_pii_middleware(
            pii_type="email",
            apply_to_input=True,
        )

        # Middleware is single instance
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_apply_to_output(self):
        """Test creating middleware with apply_to_output."""
        middleware = create_pii_middleware(
            pii_type="email",
            apply_to_output=True,
        )

        # Middleware is single instance
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_apply_to_tool_results(self):
        """Test creating middleware with apply_to_tool_results."""
        middleware = create_pii_middleware(
            pii_type="email",
            apply_to_tool_results=True,
        )

        # Middleware is single instance
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_all_apply_options(self):
        """Test creating middleware with all apply options."""
        middleware = create_pii_middleware(
            pii_type="email",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        )

        # Middleware is single instance
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_regex_string_detector(self):
        """Test creating middleware with regex string detector."""
        middleware = create_pii_middleware(
            pii_type="api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
        )

        # Middleware is single instance
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_compiled_regex_detector(self):
        """Test creating middleware with compiled regex detector."""
        pattern = re.compile(r"\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}")
        middleware = create_pii_middleware(
            pii_type="phone_number",
            detector=pattern,
            strategy="mask",
        )

        # Middleware is single instance
        assert isinstance(middleware, PIIMiddleware)

    def test_create_with_callable_detector(self):
        """Test creating middleware with callable detector."""

        def detect_custom(content: str) -> list[dict]:
            return [{"text": "test", "start": 0, "end": 4}]

        middleware = create_pii_middleware(
            pii_type="custom_type",
            detector=detect_custom,
            strategy="redact",
        )

        # Middleware is single instance
        assert isinstance(middleware, PIIMiddleware)

    def test_create_custom_type_without_detector_raises(self):
        """Test that custom type without detector raises error."""
        with pytest.raises(ValueError, match="requires a detector"):
            create_pii_middleware(pii_type="custom_type")

    def test_create_with_all_parameters(self):
        """Test creating middleware with all parameters."""
        middleware = create_pii_middleware(
            pii_type="email",
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        )

        # Middleware is single instance
        assert isinstance(middleware, PIIMiddleware)

    def test_returns_single_instance(self):
        """Test that factory returns single middleware instances."""
        email_middleware = create_pii_middleware(pii_type="email")
        card_middleware = create_pii_middleware(pii_type="credit_card")

        assert isinstance(email_middleware, PIIMiddleware)
        assert isinstance(card_middleware, PIIMiddleware)

        # Should be composable into list manually
        all_middlewares = [email_middleware, card_middleware]
        assert len(all_middlewares) == 2
        assert all(isinstance(m, PIIMiddleware) for m in all_middlewares)
