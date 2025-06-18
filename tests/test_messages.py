from pathlib import Path
from typing import Sequence

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from dao_ai.messages import message_with_images, remove_messages


@pytest.mark.unit
def test_remove_messages_without_filter() -> None:
    """Test removing all messages without a filter."""
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
        ToolMessage(content="Tool result", tool_call_id="123"),
    ]

    remove_list = remove_messages(messages)

    assert len(remove_list) == 3
    assert all(hasattr(rm, "id") for rm in remove_list)


@pytest.mark.unit
def test_remove_messages_with_filter() -> None:
    """Test removing messages with a filter function."""
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
        ToolMessage(content="Tool result", tool_call_id="123"),
    ]

    # Filter to only include HumanMessage instances
    remove_list = remove_messages(messages, lambda m: isinstance(m, HumanMessage))

    assert len(remove_list) == 1


@pytest.mark.unit
def test_remove_messages_empty_list() -> None:
    """Test removing messages from an empty list."""
    messages: Sequence[BaseMessage] = []

    remove_list = remove_messages(messages)

    assert len(remove_list) == 0


@pytest.mark.unit
def test_remove_messages_filter_excludes_all() -> None:
    """Test removing messages when filter excludes all messages."""
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]

    # Filter that excludes all messages
    remove_list = remove_messages(messages, lambda m: isinstance(m, ToolMessage))

    assert len(remove_list) == 0


@pytest.mark.unit
def test_message_with_images_single_image(tmp_path: Path) -> None:
    """Test adding a single image to a message."""
    # Create a temporary image file
    image_path = tmp_path / "test_image.png"
    image_path.write_bytes(b"fake image data")

    message = HumanMessage(content="Look at this image")

    # Test the function
    result = message_with_images(message, [image_path])

    assert isinstance(result, BaseMessage)
    assert result.content != message.content  # Should be modified


@pytest.mark.unit
def test_message_with_images_multiple_images(tmp_path: Path) -> None:
    """Test adding multiple images to a message."""
    # Create temporary image files
    image1 = tmp_path / "image1.png"
    image2 = tmp_path / "image2.jpg"
    image1.write_bytes(b"fake image data 1")
    image2.write_bytes(b"fake image data 2")

    message = HumanMessage(content="Look at these images")

    # Test the function
    result = message_with_images(message, [image1, image2])

    assert isinstance(result, BaseMessage)
    assert result.content != message.content  # Should be modified


@pytest.mark.unit
def test_message_with_images_empty_list() -> None:
    """Test adding no images to a message."""
    message = HumanMessage(content="No images here")

    # Test with empty image list
    result = message_with_images(message, [])

    assert isinstance(result, BaseMessage)
