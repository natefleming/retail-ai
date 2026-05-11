"""Tests for HumanInTheLoopModel configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dao_ai.config import HumanInTheLoopModel


class TestAllowedDecisions:
    """Tests for allowed_decisions validation in HumanInTheLoopModel."""

    def test_default_config(self):
        """Default config allows the three classic decision types (respond opt-in)."""
        model = HumanInTheLoopModel()
        assert model.allowed_decisions == ["approve", "edit", "reject"]
        assert "respond" not in model.allowed_decisions

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

    def test_respond_in_allowed_decisions(self):
        """``respond`` is now an accepted decision type (LangChain v1.x)."""
        model = HumanInTheLoopModel(allowed_decisions=["approve", "respond"])
        assert "respond" in model.allowed_decisions

    def test_all_four_decisions(self):
        """All four decision types may be enabled together."""
        model = HumanInTheLoopModel(
            allowed_decisions=["approve", "edit", "reject", "respond"]
        )
        assert model.allowed_decisions == ["approve", "edit", "reject", "respond"]

    def test_respond_only(self):
        """A respond-only config is valid (e.g. tools that should never auto-execute)."""
        model = HumanInTheLoopModel(allowed_decisions=["respond"])
        assert model.allowed_decisions == ["respond"]

    def test_respond_dedup(self):
        """Duplicate respond entries are deduplicated like every other decision."""
        model = HumanInTheLoopModel(allowed_decisions=["respond", "respond", "approve"])
        assert model.allowed_decisions == ["respond", "approve"]


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
