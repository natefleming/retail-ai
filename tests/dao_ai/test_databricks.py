import pytest
from conftest import has_databricks_env
from langchain_core.messages import BaseMessage, HumanMessage
from mlflow.pyfunc import ChatModel
from dao_ai.providers.databricks import DatabricksProvider
from dao_ai.config import AppConfig

@pytest.mark.system
@pytest.mark.slow
@pytest.mark.skipif(
    not has_databricks_env(), reason="Missing Databricks environment variables"
)
@pytest.mark.skip("Skipping Databricks agent creation test")
def test_databricks_create_agent(config: AppConfig) -> None:
    provider: DatabricksProvider = DatabricksProvider()
    provider.create_agent(config=config)
    assert True