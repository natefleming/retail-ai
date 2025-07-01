from typing import Any, Sequence

import pytest
from conftest import has_databricks_env
from langchain_core.messages import BaseMessage, HumanMessage
from mlflow.pyfunc import ChatModel

from dao_ai.models import process_messages


@pytest.mark.system
@pytest.mark.slow
@pytest.mark.skipif(
    not has_databricks_env(), reason="Missing Databricks environment variables"
)
def test_inference(chat_model: ChatModel) -> None:
    messages: Sequence[BaseMessage] = [
        HumanMessage(content="What is the weather like today?"),
    ]
    custom_inputs: dict[str, Any] = {
        "configurable": {
            "user_id": "user123",
            "thread_id": "1",
        }
    }
    response: dict[str, Any] | Any = process_messages(
        chat_model, messages, custom_inputs
    )
    print(response)
    assert response is not None


@pytest.mark.system
@pytest.mark.slow
@pytest.mark.skipif(
    not has_databricks_env(), reason="Missing Databricks environment variables"
)
def test_inference_missing_user_id(chat_model: ChatModel) -> None:
    messages: Sequence[BaseMessage] = [
        HumanMessage(content="What is the weather like today?"),
    ]
    custom_inputs: dict[str, Any] = {
        "configurable": {
            "thread_id": "1",
        }
    }
    response: dict[str, Any] | Any = process_messages(
        chat_model, messages, custom_inputs
    )
    print(response)
    assert response is not None
