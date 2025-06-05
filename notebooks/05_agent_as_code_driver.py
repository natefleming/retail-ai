# Databricks notebook source
# MAGIC %pip install --quiet uv
# MAGIC
# MAGIC import os
# MAGIC os.environ["UV_PROJECT_ENVIRONMENT"] = os.environ["VIRTUAL_ENV"]

# COMMAND ----------

# MAGIC %sh uv --project ../ sync

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="../config/model_config.yaml")
config_path: str = dbutils.widgets.get("config-path")
print(config_path)

# COMMAND ----------

import sys
from typing import Sequence
from importlib.metadata import version
from pkg_resources import get_distribution

sys.path.insert(0, "..")

pip_requirements: Sequence[str] = [
    f"databricks-agents=={version('databricks-agents')}",
    f"databricks-connect=={get_distribution('databricks-connect').version}",
    f"databricks-langchain=={version('databricks-langchain')}",
    f"databricks-sdk=={version('databricks-sdk')}",
    f"duckduckgo-search=={version('duckduckgo-search')}",
    f"langchain=={version('langchain')}",
    f"langchain-mcp-adapters=={version('langchain-mcp-adapters')}",
    f"langgraph=={version('langgraph')}",
    f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
    f"langgraph-reflection=={version('langgraph-reflection')}",
    f"langgraph-supervisor=={version('langgraph-supervisor')}",
    f"langgraph-swarm=={version('langgraph-swarm')}",
    f"langmem=={version('langmem')}",
    f"loguru=={version('loguru')}",
    f"mlflow=={version('mlflow')}",
    f"openevals=={version('openevals')}",
    f"psycopg[binary,pool]=={version('psycopg')}",
    f"pydantic=={version('pydantic')}",
    f"unitycatalog-ai[databricks]=={version('unitycatalog-ai')}",
    f"unitycatalog-langchain[databricks]=={version('unitycatalog-langchain')}",
]
print("\n".join(pip_requirements))

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from mlflow.models import ModelConfig
from retail_ai.config import AppConfig

model_config_path: str = config_path
model_config: ModelConfig = ModelConfig(development_config=model_config_path)
config: AppConfig = AppConfig(**model_config.to_dict())

# COMMAND ----------

# MAGIC %%writefile agent_as_code.py
# MAGIC import sys
# MAGIC import os
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
# MAGIC model_config_path: str = os.getenv("MODEL_CONFIG_PATH", "../config/model_config.yaml")
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

from langgraph.graph.state import CompiledStateGraph
from retail_ai.models import display_graph
from retail_ai.graph import create_retail_ai_graph

graph: CompiledStateGraph = create_retail_ai_graph(config=config)

display_graph(graph)

# COMMAND ----------

from pathlib import Path
from agent_as_code import app
from retail_ai.models import save_image

path: Path = Path.cwd().parent / Path("docs") / "architecture.png"
save_image(app, path)

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
from retail_ai.config import (
    ConnectionModel,
    FunctionModel,
    GenieRoomModel,
    IndexModel,
    LLMModel,
    TableModel,
    VectorStoreModel,
    WarehouseModel
)


llms: Sequence[LLMModel] = {llm.name: llm for llm in config.resources.llms.values()}.values()
vector_indexes: Sequence[IndexModel] = {v.index.full_name: v.index for v in config.resources.vector_stores.values()}.values()
warehouses: Sequence[WarehouseModel] = {w.warehouse_id: w for w in config.resources.warehouses.values() if w.warehouse_id}.values()
genie_rooms: Sequence[GenieRoomModel] = {g.space_id: g for g in config.resources.genie_rooms.values() if g.space_id}.values()
tables: Sequence[TableModel] = {t.full_name: t for t in config.resources.tables.values() if t.full_name}.values()
functions: Sequence[FunctionModel] = {f.full_name: f for f in config.resources.functions.values()}.values()
connections: Sequence[ConnectionModel] = {c.full_name: c for c in config.resources.connections.values()}.values()

resources: list[DatabricksResource] = []
resources += [llm.as_resource() for llm in llms]
resources += [vector_index.as_resource() for vector_index in vector_indexes]
resources += [warehouse.as_resource() for warehouse in warehouses]
resources += [genie_room.as_resource() for genie_room in genie_rooms]
resources += [function.as_resource() for function in functions]
resources += [table.as_resource() for table in tables]
resources += [connection.as_resource() for connection in connections]


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

