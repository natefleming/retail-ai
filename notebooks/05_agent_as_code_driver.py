# Databricks notebook source
# MAGIC %pip install uv
# MAGIC
# MAGIC import os
# MAGIC os.environ["UV_PROJECT_ENVIRONMENT"] = os.environ["VIRTUAL_ENV"]

# COMMAND ----------

# MAGIC %sh uv --project ../ sync

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import sys
from typing import Sequence
from importlib.metadata import version
from pkg_resources import get_distribution

sys.path.insert(0, "..")

pip_requirements: Sequence[str] = [
    f"langgraph=={version('langgraph')}",
    f"langchain=={version('langchain')}",
    f"databricks-langchain=={version('databricks-langchain')}",
    f"unitycatalog-langchain[databricks]=={version('unitycatalog-langchain')}",
    f"unitycatalog-ai[databricks]=={version('unitycatalog-ai')}",
    f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
    f"duckduckgo-search=={version('duckduckgo-search')}",
    f"databricks-sdk=={version('databricks-sdk')}",
    f"langgraph-reflection=={version('langgraph-reflection')}",
    f"langgraph-swarm=={version('langgraph-swarm')}",
    f"langgraph-supervisor=={version('langgraph-supervisor')}",
    f"openevals=={version('openevals')}",
    f"mlflow=={version('mlflow')}",
    f"psycopg[binary,pool]=={version('psycopg')}",
    f"databricks-agents=={version('databricks-agents')}",
    f"pydantic=={version('pydantic')}",
    f"loguru=={version('loguru')}",
    f"databricks-connect=={get_distribution('databricks-connect').version}"
]
print("\n".join(pip_requirements))

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %%writefile agent_as_code.py
# MAGIC import sys
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.models import ModelConfig
# MAGIC
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from mlflow.pyfunc import ChatModel
# MAGIC from retail_ai.graph import create_retail_ai_graph
# MAGIC from retail_ai.models import create_agent 
# MAGIC from retail_ai.config import AppConfig
# MAGIC
# MAGIC from loguru import logger
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC model_config_path: str = "../config/model_config.yaml"
# MAGIC model_config: ModelConfig = ModelConfig(development_config=model_config_path)
# MAGIC config: AppConfig = AppConfig(**model_config.to_dict())
# MAGIC
# MAGIC log_level: str = config.app.log_level
# MAGIC
# MAGIC logger.add(sys.stderr, level=log_level)
# MAGIC
# MAGIC graph: CompiledStateGraph = create_retail_ai_graph(config=config)
# MAGIC
# MAGIC app: ChatModel = create_agent(graph)
# MAGIC
# MAGIC mlflow.models.set_model(app)
# MAGIC

# COMMAND ----------

from agent_as_code import app
from retail_ai.models import display_graph

display_graph(app)


# COMMAND ----------

# from pathlib import Path
# from agent_as_code import app
# from retail_ai.models import save_image

# path: Path = Path("docs") / "architecture.png"
# save_image(app, path)

# COMMAND ----------

from typing import Any, Sequence, Optional

from mlflow.models.resources import (
    DatabricksResource,
    DatabricksVectorSearchIndex,
    DatabricksTable,
    DatabricksUCConnection,
    DatabricksSQLWarehouse,
    DatabricksGenieSpace,
    DatabricksFunction,
    DatabricksServingEndpoint,
)
from mlflow.models.auth_policy import (
    SystemAuthPolicy, 
    UserAuthPolicy, 
    AuthPolicy
)
import mlflow
from mlflow.models.model import ModelInfo
from agent_as_code import config


model_names: set = set([l.name for l in config.resources.llms.values()])
vector_indexes: set = set([v.index.full_name for v in config.resources.vector_stores.values()])
warehouse_ids: set = set([w.warehouse_id for w in config.resources.warehouses.values()])
space_ids: set = set([g.space_id for g in config.resources.genie_rooms.values()])
tables_names: set = set([t.full_name for t in config.resources.tables.values()])
function_names: set = set([f.full_name for f in config.resources.functions.values()])
connection_names: set = set([c.full_name for c in config.resources.connections.values()])

resources: list[DatabricksResource] = []

resources += [DatabricksServingEndpoint(endpoint_name=m) for m in model_names if m]
resources += [DatabricksVectorSearchIndex(index_name=v) for v in vector_indexes if v]
resources += [DatabricksSQLWarehouse(warehouse_id=w) for w in warehouse_ids if w]
resources += [DatabricksGenieSpace(genie_space_id=s) for s in space_ids if s]
resources += [DatabricksFunction(function_name=f) for f in function_names if f]
resources += [DatabricksTable(table_name=t) for t in tables_names if t]
resources += [DatabricksUCConnection(connection_name=c) for c in connection_names if c]

#input_example: dict[str, Any] = config.get("app").get("diy_example")

system_auth_policy: SystemAuthPolicy = SystemAuthPolicy(resources=resources)

# Api Scopes
# Vector Search:            serving.serving-endpoints, vectorsearch.vector-search-endpoints, vectorsearch.vector-search-indexes
# Model Serving Endpoints:  serving.serving-endpoints
# SQL Wareshouse:           sql.statement-execution, sql.warehouses
# UC Connections:           catalog.connections
# Genie:                    dashboards.genie

user_auth_policy: UserAuthPolicy = UserAuthPolicy(
    api_scopes=[
        "serving.serving-endpoints",
        "vectorsearch.vector-search-endpoints",
        "vectorsearch.vector-search-indexes",
        "sql.statement-execution", 
        "sql.warehouses",
        "catalog.connections",
        "dashboards.genie",
    ]
)
auth_policy: AuthPolicy = AuthPolicy(
    system_auth_policy=system_auth_policy,
    user_auth_policy=user_auth_policy
)

with mlflow.start_run(run_name="agent"):
    mlflow.set_tag("type", "agent")
    logged_agent_info: ModelInfo = mlflow.pyfunc.log_model(
        python_model="agent_as_code.py",
        code_paths=["../retail_ai"],
        model_config=config.model_dump(),
        artifact_path="agent",
        pip_requirements=pip_requirements,
        resources=resources,
        #auth_policy=auth_policy,
    )

# COMMAND ----------

import mlflow
from mlflow.entities.model_registry.model_version import ModelVersion
from retail_ai.catalog import full_name
from agent_as_code import config


mlflow.set_registry_uri("databricks-uc")

registered_model_name: str = config.app.registered_model.full_name

model_version: ModelVersion = mlflow.register_model(
    name=registered_model_name, 
    model_uri=logged_agent_info.model_uri
)

# COMMAND ----------

from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion

client: MlflowClient = MlflowClient()

client.set_registered_model_alias(
    name=registered_model_name, alias="Champion", version=model_version.version
)
champion_model: ModelVersion = client.get_model_version_by_alias(
    registered_model_name, "Champion"
)
print(champion_model)

# COMMAND ----------

from databricks import agents
from retail_ai.models import get_latest_model_version
from agent_as_code import config


registered_model_name: str = config.app.registered_model.full_name
endpoint_name: str = config.app.endpoint_name
tags: dict[str, str] = config.app.tags
latest_version: int = get_latest_model_version(registered_model_name)

agents.deploy(
    model_name=registered_model_name,
    model_version=latest_version,
    scale_to_zero=True,
    environment_vars={},
    workload_size="Small",
    endpoint_name=endpoint_name,
    tags=tags,
)

# COMMAND ----------

from typing import Any, Sequence
from retail_ai.catalog import full_name
from databricks.agents import set_permissions, PermissionLevel
from rich import print as pprint
from agent_as_code import config

registered_model_name: str = config.app.registered_model.full_name
permissions: Sequence[dict[str, Any]] = config.app.permissions

pprint(registered_model_name)
pprint(permissions)

for permission in permissions:
    principals: Sequence[str] = permission.principals
    entitlements: Sequence[str] = permission.entitlements

    if not principals or not entitlements:
        continue
    for entitlement in entitlements:
        set_permissions(
            model_name=registered_model_name,
            users=principals,
            permission_level=PermissionLevel[entitlement]
        )


# COMMAND ----------

from typing import Any
from agent_as_code import config

example_input: dict[str, Any] = config.get("app").get("diy_example")

mlflow.models.predict(
    model_uri=logged_agent_info.model_uri,
    input_data=example_input,
    env_manager="uv",
)

# COMMAND ----------

from typing import Any
from mlflow.deployments import get_deploy_client
from rich import print as pprint
from agent_as_code import config

endpoint_name: str = config.get("app").get("endpoint_name")
example_input: dict[str, Any] = config.get("app").get("inventory_example")

response = get_deploy_client("databricks").predict(
    endpoint=endpoint_name,
    inputs=example_input,
)

pprint(response)
