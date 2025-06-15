from typing import Sequence

import pytest
from conftest import has_databricks_env
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.pregel.io import AddableValuesDict
from mlflow.pyfunc import ChatModel

from retail_ai.models import process_messages


@pytest.mark.system
@pytest.mark.slow
@pytest.mark.skipif(
    not has_databricks_env(), reason="Missing Databricks environment variables"
)
def test_inference(chat_model: ChatModel) -> None:
    messages: Sequence[BaseMessage] = [
        HumanMessage(content="What is the weather like today?"),
    ]
    response: AddableValuesDict = process_messages(chat_model, messages)
    print(response)
    assert response is not None
