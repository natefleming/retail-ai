"""Test to verify LangGraph Interrupt type structure."""

from typing import Any
from unittest.mock import MagicMock

from langgraph.types import Interrupt, StateSnapshot

from dao_ai.models import is_interrupted


class TestInterruptType:
    """Test the LangGraph Interrupt type."""

    def test_interrupt_has_value_attribute(self):
        """Verify Interrupt has a value attribute."""
        test_value = {"action_requests": [], "review_configs": []}
        interrupt = Interrupt(value=test_value, id="test-id")

        assert hasattr(interrupt, "value")
        assert interrupt.value == test_value

    def test_interrupt_has_id_attribute(self):
        """Verify Interrupt has an id attribute."""
        interrupt = Interrupt(value={}, id="test-123")

        assert hasattr(interrupt, "id")
        assert interrupt.id == "test-123"

    def test_interrupt_value_can_be_dict(self):
        """Verify Interrupt value can be a dictionary."""
        test_dict = {
            "action_requests": [
                {
                    "name": "test_tool",
                    "args": {"arg1": "value1"},
                    "description": "Test action",
                }
            ],
            "review_configs": [
                {
                    "action_name": "test_tool",
                    "allowed_decisions": ["approve", "reject"],
                }
            ],
        }
        interrupt = Interrupt(value=test_dict, id="test-id")

        assert isinstance(interrupt.value, dict)
        assert "action_requests" in interrupt.value
        assert "review_configs" in interrupt.value

    def test_interrupt_value_type_is_any(self):
        """Verify Interrupt value can be any type."""
        # Test with dict
        interrupt1 = Interrupt(value={"key": "value"}, id="id1")
        assert interrupt1.value == {"key": "value"}

        # Test with string
        interrupt2 = Interrupt(value="string value", id="id2")
        assert interrupt2.value == "string value"

        # Test with list
        interrupt3 = Interrupt(value=[1, 2, 3], id="id3")
        assert interrupt3.value == [1, 2, 3]

    def test_interrupt_type_annotation(self):
        """Verify Interrupt can be used in type annotations."""
        from langgraph.types import Interrupt

        def process_interrupt(interrupt: Interrupt) -> dict[str, Any]:
            """Example function with Interrupt type hint."""
            return interrupt.value if isinstance(interrupt.value, dict) else {}

        test_interrupt = Interrupt(value={"test": "data"}, id="test")
        result = process_interrupt(test_interrupt)

        assert result == {"test": "data"}

    def test_interrupt_in_list(self):
        """Verify list of Interrupts works correctly."""
        interrupts: list[Interrupt] = [
            Interrupt(value={"first": 1}, id="id1"),
            Interrupt(value={"second": 2}, id="id2"),
        ]

        assert len(interrupts) == 2
        assert interrupts[0].value == {"first": 1}
        assert interrupts[1].value == {"second": 2}


class TestIsInterrupted:
    """Test the is_interrupted utility function."""

    def test_is_interrupted_with_empty_tuple(self):
        """Verify is_interrupted returns False when interrupts tuple is empty."""
        # Create a mock StateSnapshot with no interrupts
        snapshot = MagicMock(spec=StateSnapshot)
        snapshot.interrupts = ()  # Empty tuple = no interrupts

        assert is_interrupted(snapshot) is False

    def test_is_interrupted_with_single_interrupt(self):
        """Verify is_interrupted returns True when there is one interrupt."""
        # Create a mock StateSnapshot with one interrupt
        snapshot = MagicMock(spec=StateSnapshot)
        snapshot.interrupts = (
            Interrupt(
                value={
                    "action_requests": [{"name": "send_email", "args": {}}],
                    "review_configs": [{"action_name": "send_email"}],
                },
                id="interrupt-1",
            ),
        )

        assert is_interrupted(snapshot) is True

    def test_is_interrupted_with_multiple_interrupts(self):
        """Verify is_interrupted returns True when there are multiple interrupts."""
        # Create a mock StateSnapshot with multiple interrupts
        snapshot = MagicMock(spec=StateSnapshot)
        snapshot.interrupts = (
            Interrupt(value={"action_requests": []}, id="interrupt-1"),
            Interrupt(value={"action_requests": []}, id="interrupt-2"),
        )

        assert is_interrupted(snapshot) is True
        assert len(snapshot.interrupts) == 2

    def test_is_interrupted_follows_langchain_pattern(self):
        """Verify is_interrupted follows LangChain documentation pattern.

        From LangChain docs, StateSnapshot.interrupts is a tuple:
        - Empty tuple () when not interrupted
        - Contains Interrupt objects when interrupted
        """
        # Not interrupted case
        not_interrupted = MagicMock(spec=StateSnapshot)
        not_interrupted.interrupts = ()
        assert is_interrupted(not_interrupted) is False

        # Interrupted case
        interrupted = MagicMock(spec=StateSnapshot)
        interrupted.interrupts = (
            Interrupt(
                value={
                    "action_requests": [
                        {
                            "name": "execute_sql",
                            "args": {"query": "DELETE FROM records"},
                        }
                    ],
                    "review_configs": [
                        {
                            "action_name": "execute_sql",
                            "allowed_decisions": ["approve", "edit", "reject"],
                        }
                    ],
                },
                id="hitl-interrupt",
            ),
        )
        assert is_interrupted(interrupted) is True
