"""Tests for HumanInTheLoopModel configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dao_ai.config import HumanInTheLoopModel


class TestAllowedDecisions:
    """Tests for allowed_decisions validation in HumanInTheLoopModel."""

    def test_default_config(self):
        """Test that default config allows all three decision types."""
        model = HumanInTheLoopModel()
        assert model.allowed_decisions == ["approve", "edit", "reject"]

    def test_custom_allowed_decisions(self):
        """Test setting custom allowed decisions."""
        model = HumanInTheLoopModel(allowed_decisions=["approve", "reject"])
        assert model.allowed_decisions == ["approve", "reject"]
        assert "edit" not in model.allowed_decisions

    def test_single_decision_type(self):
        """Test configuring only one decision type."""
        model = HumanInTheLoopModel(allowed_decisions=["approve"])
        assert model.allowed_decisions == ["approve"]

    def test_removes_duplicates(self):
        """Test that duplicate decisions are removed."""
        model = HumanInTheLoopModel(
            allowed_decisions=["approve", "approve", "edit", "reject", "edit"]
        )
        assert model.allowed_decisions == ["approve", "edit", "reject"]

    def test_empty_decisions_raises_error(self):
        """Test that empty allowed_decisions raises validation error."""
        with pytest.raises(
            ValidationError, match="At least one decision type must be allowed"
        ):
            HumanInTheLoopModel(allowed_decisions=[])

    def test_invalid_decision_type_raises_error(self):
        """Test that invalid decision types raise validation error."""
        with pytest.raises(ValidationError):
            HumanInTheLoopModel(allowed_decisions=["invalid_decision"])


class TestHumanInTheLoopModel:
    """Tests for HumanInTheLoopModel."""

    def test_default_model(self):
        """Test default HumanInTheLoopModel configuration."""
        model = HumanInTheLoopModel()
        assert model.review_prompt is None
        assert model.allowed_decisions == ["approve", "edit", "reject"]

    def test_new_format_with_allowed_decisions(self):
        """Test new format using allowed_decisions directly."""
        model = HumanInTheLoopModel(
            review_prompt="Custom review prompt",
            allowed_decisions=["approve", "reject"],
        )
        assert model.review_prompt == "Custom review prompt"
        assert model.allowed_decisions == ["approve", "reject"]

    def test_new_format_with_allowed_decisions_explicit(self):
        """Test new format using allowed_decisions list directly."""
        model = HumanInTheLoopModel(
            review_prompt="Review only",
            allowed_decisions=["approve"],
        )
        assert model.allowed_decisions == ["approve"]

    def test_all_decisions_enabled(self):
        """Test with all decision types enabled."""
        model = HumanInTheLoopModel(allowed_decisions=["approve", "edit", "reject"])
        assert "approve" in model.allowed_decisions
        assert "edit" in model.allowed_decisions
        assert "reject" in model.allowed_decisions

    def test_only_approve_enabled(self):
        """Test with only approve decision enabled."""
        model = HumanInTheLoopModel(allowed_decisions=["approve"])
        assert model.allowed_decisions == ["approve"]

    def test_only_edit_enabled(self):
        """Test with only edit decision enabled."""
        model = HumanInTheLoopModel(allowed_decisions=["edit"])
        assert model.allowed_decisions == ["edit"]

    def test_approve_and_reject_enabled(self):
        """Test with approve and reject decisions enabled."""
        model = HumanInTheLoopModel(allowed_decisions=["approve", "reject"])
        assert model.allowed_decisions == ["approve", "reject"]

    def test_model_serialization(self):
        """Test that model can be serialized and deserialized."""
        original = HumanInTheLoopModel(
            review_prompt="Test prompt",
            allowed_decisions=["approve", "reject"],
        )

        # Serialize to dict
        data = original.model_dump()

        # Deserialize from dict
        restored = HumanInTheLoopModel(**data)

        assert restored.review_prompt == original.review_prompt
        assert restored.allowed_decisions == ["approve", "reject"]
