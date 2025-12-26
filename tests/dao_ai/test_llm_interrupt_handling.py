"""Test LLM-based interrupt handling for Human-in-the-Loop."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage
from langgraph.types import Interrupt, StateSnapshot
from pydantic import BaseModel

from dao_ai.models import (
    _convert_schema_to_decisions,
    _create_decision_schema,
    handle_interrupt_response,
)


class TestCreateDecisionSchema:
    """Test dynamic schema creation for interrupt decisions."""

    def test_creates_schema_for_single_action(self):
        """Verify schema is created correctly for a single action."""
        interrupt_data = [
            {
                "action_requests": [
                    {
                        "name": "send_email",
                        "args": {"to": "user@example.com"},
                    }
                ],
                "review_configs": [
                    {
                        "action_name": "send_email",
                        "allowed_decisions": ["approve", "reject"],
                    }
                ],
            }
        ]

        DecisionsModel = _create_decision_schema(interrupt_data)

        # Verify model has correct fields
        assert hasattr(DecisionsModel, "model_fields")
        fields = DecisionsModel.model_fields
        assert "decision_1" in fields
        assert "decision_1_message" in fields  # For reject

    def test_creates_schema_for_multiple_actions(self):
        """Verify schema handles multiple actions correctly."""
        interrupt_data = [
            {
                "action_requests": [
                    {"name": "send_email", "args": {}},
                    {"name": "execute_sql", "args": {"query": "SELECT *"}},
                ],
                "review_configs": [
                    {
                        "action_name": "send_email",
                        "allowed_decisions": ["approve", "edit", "reject"],
                    },
                    {
                        "action_name": "execute_sql",
                        "allowed_decisions": ["approve", "reject"],
                    },
                ],
            }
        ]

        DecisionsModel = _create_decision_schema(interrupt_data)

        fields = DecisionsModel.model_fields
        assert "decision_1" in fields
        assert "decision_1_message" in fields  # For reject
        assert "decision_1_edited_args" in fields  # For edit
        assert "decision_2" in fields
        assert "decision_2_message" in fields

    def test_schema_includes_only_allowed_decisions(self):
        """Verify schema only includes fields for allowed decision types."""
        interrupt_data = [
            {
                "action_requests": [{"name": "read_file", "args": {}}],
                "review_configs": [
                    {
                        "action_name": "read_file",
                        "allowed_decisions": ["approve"],  # Only approve allowed
                    }
                ],
            }
        ]

        DecisionsModel = _create_decision_schema(interrupt_data)

        fields = DecisionsModel.model_fields
        assert "decision_1" in fields
        # Should not have message or edited_args since only approve is allowed
        assert "decision_1_message" not in fields
        assert "decision_1_edited_args" not in fields


class TestConvertSchemaToDecisions:
    """Test conversion from parsed schema to Decision objects."""

    def test_converts_approve_decision(self):
        """Verify approve decision is converted correctly."""

        # Create a simple model for testing
        class TestDecisions(BaseModel):
            decision_1: str = "approve"

        parsed = TestDecisions()
        interrupt_data = [
            {
                "action_requests": [{"name": "send_email", "args": {}}],
                "review_configs": [],
            }
        ]

        decisions = _convert_schema_to_decisions(parsed, interrupt_data)

        assert len(decisions) == 1
        assert decisions[0]["type"] == "approve"

    def test_converts_reject_decision_with_message(self):
        """Verify reject decision with message is converted correctly."""

        class TestDecisions(BaseModel):
            decision_1: str = "reject"
            decision_1_message: str = "Too risky"

        parsed = TestDecisions()
        interrupt_data = [
            {
                "action_requests": [{"name": "execute_sql", "args": {}}],
                "review_configs": [],
            }
        ]

        decisions = _convert_schema_to_decisions(parsed, interrupt_data)

        assert len(decisions) == 1
        assert decisions[0]["type"] == "reject"
        assert decisions[0]["message"] == "Too risky"

    def test_converts_edit_decision(self):
        """Verify edit decision merges args correctly."""

        class TestDecisions(BaseModel):
            decision_1: str = "edit"
            decision_1_edited_args: dict = {"to": "new@example.com"}

        parsed = TestDecisions()
        interrupt_data = [
            {
                "action_requests": [
                    {
                        "name": "send_email",
                        "args": {"to": "old@example.com", "subject": "Test"},
                    }
                ],
                "review_configs": [],
            }
        ]

        decisions = _convert_schema_to_decisions(parsed, interrupt_data)

        assert len(decisions) == 1
        assert decisions[0]["type"] == "edit"
        assert decisions[0]["edited_action"]["name"] == "send_email"
        # Should merge original and edited args
        assert decisions[0]["edited_action"]["args"]["to"] == "new@example.com"
        assert decisions[0]["edited_action"]["args"]["subject"] == "Test"

    def test_converts_multiple_decisions(self):
        """Verify multiple decisions are converted in order."""

        class TestDecisions(BaseModel):
            decision_1: str = "approve"
            decision_2: str = "reject"
            decision_2_message: str = "Not authorized"

        parsed = TestDecisions()
        interrupt_data = [
            {
                "action_requests": [
                    {"name": "send_email", "args": {}},
                    {"name": "execute_sql", "args": {}},
                ],
                "review_configs": [],
            }
        ]

        decisions = _convert_schema_to_decisions(parsed, interrupt_data)

        assert len(decisions) == 2
        assert decisions[0]["type"] == "approve"
        assert decisions[1]["type"] == "reject"
        assert decisions[1]["message"] == "Not authorized"


class TestHandleInterruptResponse:
    """Test LLM-based interrupt response handling."""

    @patch("dao_ai.models.ChatDatabricks")
    def test_parses_approve_message(self, mock_chat):
        """Verify LLM parses approval message correctly."""
        # Setup mock LLM response
        mock_llm = MagicMock()
        mock_structured = MagicMock()

        # Create a mock parsed response
        class MockDecisions(BaseModel):
            is_valid: bool = True
            validation_message: str | None = None
            decision_1: str = "approve"

        mock_structured.invoke.return_value = MockDecisions()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat.return_value = mock_llm

        # Create mock snapshot with interrupt
        snapshot = MagicMock(spec=StateSnapshot)
        interrupt = MagicMock(spec=Interrupt)
        interrupt.value = {
            "action_requests": [{"name": "send_email", "args": {}}],
            "review_configs": [
                {
                    "action_name": "send_email",
                    "allowed_decisions": ["approve", "reject"],
                }
            ],
        }
        snapshot.interrupts = (interrupt,)

        # Call handler
        messages = [HumanMessage(content="Yes, go ahead and send it")]
        result = handle_interrupt_response(
            snapshot=snapshot,
            messages=messages,
            model=mock_llm,
        )

        # Verify result
        assert result["is_valid"] is True
        assert result["validation_message"] is None
        assert "decisions" in result
        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["type"] == "approve"

    @patch("dao_ai.models.ChatDatabricks")
    def test_parses_reject_message(self, mock_chat):
        """Verify LLM parses rejection message correctly."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()

        class MockDecisions(BaseModel):
            is_valid: bool = True
            validation_message: str | None = None
            decision_1: str = "reject"
            decision_1_message: str = "Too risky"

        mock_structured.invoke.return_value = MockDecisions()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat.return_value = mock_llm

        snapshot = MagicMock(spec=StateSnapshot)
        interrupt = MagicMock(spec=Interrupt)
        interrupt.value = {
            "action_requests": [{"name": "execute_sql", "args": {"query": "DELETE *"}}],
            "review_configs": [
                {
                    "action_name": "execute_sql",
                    "allowed_decisions": ["approve", "reject"],
                }
            ],
        }
        snapshot.interrupts = (interrupt,)

        messages = [HumanMessage(content="No, don't run that query - too risky")]
        result = handle_interrupt_response(
            snapshot=snapshot,
            messages=messages,
            model=mock_llm,
        )

        assert result["is_valid"] is True
        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["type"] == "reject"
        assert result["decisions"][0]["message"] == "Too risky"

    @patch("dao_ai.models.ChatDatabricks")
    def test_handles_empty_interrupts(self, mock_chat):
        """Verify handler gracefully handles no interrupts."""
        snapshot = MagicMock(spec=StateSnapshot)
        snapshot.interrupts = ()

        messages = [HumanMessage(content="Yes")]
        result = handle_interrupt_response(
            snapshot=snapshot,
            messages=messages,
            model=None,
        )

        assert result["decisions"] == []

    @patch("dao_ai.models.ChatDatabricks")
    def test_handles_llm_parsing_failure(self, mock_chat):
        """Verify handler returns invalid response on parsing failure."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("LLM error")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat.return_value = mock_llm

        snapshot = MagicMock(spec=StateSnapshot)
        interrupt = MagicMock(spec=Interrupt)
        interrupt.value = {
            "action_requests": [
                {"name": "send_email", "args": {}},
                {"name": "execute_sql", "args": {}},
            ],
            "review_configs": [],
        }
        snapshot.interrupts = (interrupt,)

        messages = [HumanMessage(content="Some message")]
        result = handle_interrupt_response(
            snapshot=snapshot,
            messages=messages,
            model=mock_llm,
        )

        # Should return invalid response with empty decisions
        assert result["is_valid"] is False
        assert "validation_message" in result
        assert len(result["decisions"]) == 0
        assert "Failed to parse" in result["validation_message"]

    def test_handles_no_human_message(self):
        """Verify handler returns invalid response when no human message is provided."""
        snapshot = MagicMock(spec=StateSnapshot)
        interrupt = MagicMock(spec=Interrupt)
        interrupt.value = {
            "action_requests": [{"name": "send_email", "args": {}}],
            "review_configs": [],
        }
        snapshot.interrupts = (interrupt,)

        # Empty message list or only AI messages
        messages = []
        result = handle_interrupt_response(
            snapshot=snapshot,
            messages=messages,
            model=None,
        )

        # Should return invalid response
        assert result["is_valid"] is False
        assert "validation_message" in result
        assert len(result["decisions"]) == 0
        assert "No user message found" in result["validation_message"]
