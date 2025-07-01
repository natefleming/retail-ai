import pytest
from conftest import has_databricks_env

from dao_ai.config import AppConfig
from dao_ai.providers.databricks import DatabricksProvider


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
