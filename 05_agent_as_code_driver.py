# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "langgraph",
  "langchain",
  "databricks-langchain",
  "langgraph-checkpoint-postgres",
  "databricks-agents",
  "psycopg[binary,pool]", 
  "databricks-sdk",
  "mlflow",
  "pydantic",
  "python-dotenv",
  "uv",
  "grandalf",
)

pip_requirements: str = " ".join(pip_requirements)

%pip install --quiet --upgrade {pip_requirements}
%restart_python

# COMMAND ----------

from importlib.metadata import version

from typing import Sequence

from importlib.metadata import version

pip_requirements: Sequence[str] = [
    f"langgraph=={version('langgraph')}",
    f"langchain=={version('langchain')}",
    f"databricks-langchain=={version('databricks-langchain')}",
    f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
    f"databricks-sdk=={version('databricks-sdk')}",
    f"mlflow=={version('mlflow')}",
    f"psycopg[binary,pool]=={version('psycopg')}", 
    f"databricks-agents=={version('databricks-agents')}",
    f"pydantic=={version('pydantic')}",
]
print("\n".join(pip_requirements))


# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %%writefile agent_as_code.py
# MAGIC
# MAGIC from typing import Sequence
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.models import ModelConfig
# MAGIC
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC
# MAGIC from retail_ai.graph import create_graph
# MAGIC from retail_ai.models import LangGraphChatAgent, create_agent
# MAGIC
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC config: ModelConfig = ModelConfig(development_config="model_config.yaml")
# MAGIC
# MAGIC model_name: str = config.get("llms").get("model_name")
# MAGIC
# MAGIC endpoint: str = config.get("retriever").get("endpoint_name")
# MAGIC index_name: str = config.get("retriever").get("index_name")
# MAGIC search_parameters: dict[str, str] = config.get("retriever").get("search_parameters")
# MAGIC primary_key: str = config.get("retriever").get("primary_key")
# MAGIC doc_uri: str = config.get("retriever").get("doc_uri")
# MAGIC text_column: str = config.get("retriever").get("embedding_source_column")
# MAGIC columns: Sequence[str] = config.get("retriever").get("columns")
# MAGIC space_id: str = config.get("genie").get("space_id")
# MAGIC
# MAGIC
# MAGIC graph: CompiledStateGraph = (
# MAGIC     create_graph(
# MAGIC         model_name=model_name,
# MAGIC         endpoint=endpoint,
# MAGIC         index_name=index_name,
# MAGIC         primary_key=primary_key,
# MAGIC         doc_uri=doc_uri,
# MAGIC         text_column=text_column,
# MAGIC         columns=columns,
# MAGIC         search_parameters=search_parameters,
# MAGIC         space_id=space_id
# MAGIC     )
# MAGIC )
# MAGIC app: LangGraphChatAgent = create_agent(graph)
# MAGIC
# MAGIC
# MAGIC mlflow.models.set_model(app)
# MAGIC

# COMMAND ----------

from agent_as_code import graph

from IPython.display import HTML, Image, display


# try:
#   content = Image(graph.get_graph(xray=True).draw_mermaid_png())
# except Exception as e:
ascii_graph: str = graph.get_graph(xray=True).draw_ascii()
html_content = f"""
<pre style="font-family: monospace; line-height: 1.2; white-space: pre;">
{ascii_graph}
</pre>
"""
content = HTML(html_content)

display(content)

# COMMAND ----------

from typing import Any
from agent_as_code import app, config

example_input: dict[str, Any] = config.get("app").get("inventory_example")

app.predict(example_input)

# COMMAND ----------

from typing import Any
from agent_as_code import app, config

example_input: dict[str, Any] = config.get("app").get("recommendation_example")

app.predict(example_input)

# COMMAND ----------

from typing import Any
from mlflow.types.agent import ChatAgentResponse
from retail_ai.models import process_messages
from agent_as_code import app, config

example_input: dict[str, Any] = config.get("app").get("example_input")

result: ChatAgentResponse = process_messages(app, example_input)
print(result.messages[-1].content)
print(result.usage)

# COMMAND ----------

from typing import Any
from mlflow.types.agent import ChatAgentChunk
from retail_ai.models import process_messages_stream
from agent_as_code import app, config


example_input: dict[str, Any] = config.get("app").get("example_input")

for event in process_messages_stream(app, example_input["messages"]):
  event: ChatAgentChunk
  print(event, "-----------\n")


# COMMAND ----------


from typing import Sequence

from databricks_langchain import VectorSearchRetrieverTool

from mlflow.models.resources import (
    DatabricksResource,
    DatabricksVectorSearchIndex,
    DatabricksTable,
    DatabricksUCConnection,
    DatabricksSQLWarehouse,
    DatabricksGenieSpace, 
    DatabricksFunction, 
    DatabricksServingEndpoint
)
import mlflow
from mlflow.models.model import ModelInfo

from agent_as_code import config


model_name: str = config.get("llms").get("model_name")
index_name: str = config.get("retriever").get("index_name")
space_id: str = config.get("genie").get("space_id")
functions: Sequence[str] = config.get("functions")

resources: Sequence[DatabricksResource] = [
    DatabricksServingEndpoint(endpoint_name=model_name),
    DatabricksVectorSearchIndex(index_name=index_name),
    DatabricksGenieSpace(genie_space_id=space_id),
]
resources += [DatabricksFunction(function_name=f) for f in functions]


with mlflow.start_run(run_name="agent"):
    mlflow.set_tag("type", "agent")
    logged_agent_info: ModelInfo = mlflow.pyfunc.log_model(
        python_model="agent_as_code.py",
        code_paths=["retail_ai"],
        model_config=config.to_dict(),
        artifact_path="agent",
        pip_requirements=pip_requirements,
        resources=resources,
    )

# COMMAND ----------

from typing import Any
from agent_as_code import config

example_input: dict[str, Any] = config.get("app").get("example_input")

mlflow.models.predict(
    model_uri=logged_agent_info.model_uri,
    input_data=example_input,
    env_manager="uv",
)

# COMMAND ----------

import mlflow
from mlflow.models.model import ModelInfo
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.models.evaluation import EvaluationResult

import pandas as pd

from agent_as_code import config


model_info: mlflow.models.model.ModelInfo
evaluation_result: EvaluationResult

evaluation_table_name: str = config.get("evaluation").get("table_name")

evaluation_pdf: pd.DataFrame = spark.table(evaluation_table_name).toPandas()

global_guidelines = {
  "English": ["The response must be in English"],
  "Clarity": ["The response must be clear, coherent, and concise"],
}

with mlflow.start_run():
  eval_results = mlflow.evaluate(
      data=evaluation_pdf,            
      model=logged_agent_info.model_uri,    
      model_type="databricks-agent",  
      evaluator_config={
        "databricks-agent": {"global_guidelines": global_guidelines}
      }
  )


# COMMAND ----------

import mlflow
from mlflow.entities.model_registry.model_version import ModelVersion

from agent_as_code import config


mlflow.set_registry_uri("databricks-uc")

registered_model_name: str = config.get("app").get("registered_model_name")

model_version: ModelVersion = mlflow.register_model(
    name=registered_model_name,
    model_uri=logged_agent_info.model_uri
)


# COMMAND ----------

from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion

client: MlflowClient = MlflowClient()

client.set_registered_model_alias(name=registered_model_name, alias="Champion", version=model_version.version)
champion_model: ModelVersion = client.get_model_version_by_alias(registered_model_name, "Champion")
print(champion_model)

# COMMAND ----------

from databricks import agents
from databricks.sdk.service.serving import (
    ServedModelInputWorkloadSize, 
)

from agent_as_code import config
from retail_ai.models import get_latest_model_version

registered_model_name: str = config.get("app").get("registered_model_name")
endpoint_name: str = config.get("app").get("endpoint_name")
tags: dict[str, str] = config.get("app").get("tags")
latest_version: int = get_latest_model_version(registered_model_name)


agents.deploy(
  model_name=registered_model_name, 
  model_version=latest_version, 
  scale_to_zero=True,
  environment_vars={},
  workload_size=ServedModelInputWorkloadSize.SMALL,
  endpoint_name=endpoint_name,
  tags=tags
)

# COMMAND ----------

from typing import Sequence

from databricks.agents import set_permissions, PermissionLevel

from agent_as_code import config


users: Sequence[str] = config.get("app").get("users") 

if users:
  set_permissions(model_name=registered_model_name, users=users, permission_level=PermissionLevel.CAN_MANAGE)


# COMMAND ----------

from typing import Any
from mlflow.deployments import get_deploy_client

from agent_as_code import config

endpoint_name: str = config.get("app").get("endpoint_name")
example_input: dict[str, Any] = config.get("app").get("example_input")

response = get_deploy_client("databricks").predict(
  endpoint=endpoint_name,
  inputs=example_input,
)

print(response["messages"][-1]["content"])

# COMMAND ----------

example_input

# COMMAND ----------

from typing import Any
from mlflow.deployments import get_deploy_client

from agent_as_code import config
from rich import print as pprint


endpoint_name: str = config.get("app").get("endpoint_name")
example_input: dict[str, Any] = config.get("app").get("example_input")

response = get_deploy_client("databricks").predict(
  endpoint=endpoint_name,
  inputs=example_input,
)

pprint(response["messages"])
