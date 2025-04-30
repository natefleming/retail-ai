# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "langgraph",
  "langchain",
  "databricks-langchain",
  "unitycatalog-langchain[databricks]",
  "langgraph-checkpoint-postgres",
  "databricks-agents",
  "psycopg[binary,pool]", 
  "databricks-sdk",
  "mlflow",
  "pydantic",
  "python-dotenv",
  "uv",
  "grandalf",
  "loguru",
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
    f"unitycatalog-langchain[databricks]=={version('unitycatalog-langchain')}",
    f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
    f"databricks-sdk=={version('databricks-sdk')}",
    f"mlflow=={version('mlflow')}",
    f"psycopg[binary,pool]=={version('psycopg')}", 
    f"databricks-agents=={version('databricks-agents')}",
    f"pydantic=={version('pydantic')}",
    f"loguru=={version('loguru')}",
]
print("\n".join(pip_requirements))


# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %%writefile agent_as_code.py
# MAGIC
# MAGIC from typing import Sequence
# MAGIC import sys
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.models import ModelConfig
# MAGIC
# MAGIC from langchain_core.runnables import RunnableSequence
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC
# MAGIC from retail_ai.graph import create_retail_ai_graph
# MAGIC from retail_ai.models import LangGraphChatAgent, create_agent, as_langgraph_chain
# MAGIC
# MAGIC from loguru import logger
# MAGIC
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
# MAGIC log_level: str = config.get("app").get("log_level")
# MAGIC
# MAGIC logger.add(sys.stderr, level=log_level)
# MAGIC
# MAGIC graph: CompiledStateGraph = (
# MAGIC     create_retail_ai_graph(
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
# MAGIC
# MAGIC #app: LangGraphChatAgent = create_agent(graph)
# MAGIC
# MAGIC app: RunnableSequence = as_langgraph_chain(graph)
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

app.invoke(example_input)

# COMMAND ----------

example_input

# COMMAND ----------

from typing import Any
from agent_as_code import app, config

example_input: dict[str, Any] = config.get("app").get("recommendation_example")

app.invoke(example_input)

# COMMAND ----------


from typing import Sequence, Optional

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
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    StringResponse,
)
from dataclasses import dataclass, field, asdict
from agent_as_code import config


model_name: str = config.get("llms").get("model_name")
index_name: str = config.get("retriever").get("index_name")
space_id: str = config.get("genie").get("space_id")
functions: Sequence[str] = config.get("functions")
tables: Sequence[str] = config.get("tables")

@dataclass
class ConfigurableInputs():
    thread_id: str = None
    user_id: str = None
    scd_ids: Optional[list[str]] = field(default_factory=list)
    store_num: int = None


# Additional input fields must be marked as Optional and have a default value
@dataclass
class CustomChatCompletionRequest(ChatCompletionRequest):
    configurable: Optional[dict[str, Any]] = field(default_factory={})
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None

sample_input = CustomChatCompletionRequest(
    messages=[{"role": "user", "content": "What is the inventory of the product with id 1?"}],
    configurable={
        "thread_id": None,
        "user_id": None,
        "scd_ids": None,
        "store_num": None,
    },
    temperature=None,
    max_tokens=None,
    stream=None

)

signature: ModelSignature = infer_signature(asdict(sample_input), StringResponse())


resources: Sequence[DatabricksResource] = [
    DatabricksServingEndpoint(endpoint_name=model_name),
    DatabricksVectorSearchIndex(index_name=index_name),
    DatabricksGenieSpace(genie_space_id=space_id),
]
resources += [DatabricksFunction(function_name=f) for f in functions]
resources += [DatabricksTable(table_name=t) for t in tables]

input_example: dict[str, Any] = config.get("app").get("example_input")

with mlflow.start_run(run_name="agent"):
    mlflow.set_tag("type", "agent")
    logged_agent_info: ModelInfo = mlflow.langchain.log_model(
        lc_model="agent_as_code.py",
        code_paths=["retail_ai"],
        model_config=config.to_dict(),
        artifact_path="agent",
        pip_requirements=pip_requirements,
        resources=resources,
        signature=signature,
        input_example=input_example,
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

from rich import print as pprint


input_example: dict[str, Any] = config.get("app").get("example_input")
pprint(input_example)

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
from rich import print as pprint
from agent_as_code import config

endpoint_name: str = config.get("app").get("endpoint_name")
example_input: dict[str, Any] = config.get("app").get("inventory_example")

response = get_deploy_client("databricks").predict(
  endpoint=endpoint_name,
  inputs=example_input,
)

pprint(response)

# COMMAND ----------

response

# COMMAND ----------

from typing import Any
from mlflow.deployments import get_deploy_client

from agent_as_code import config
from rich import print as pprint


endpoint_name: str = config.get("app").get("endpoint_name")
example_input: dict[str, Any] = config.get("app").get("recommendation_example")

response = get_deploy_client("databricks").predict(
  endpoint=endpoint_name,
  inputs=example_input,
)

pprint(response)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from rich import print as pprint

w: WorkspaceClient = WorkspaceClient()

openai_client = w.serving_endpoints.get_open_ai_client()

response = openai_client.chat.completions.create(
    model=endpoint_name,
    messages=example_input["messages"],
)

pprint(response)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from rich import print as pprint


w: WorkspaceClient = WorkspaceClient()

openai_client = w.serving_endpoints.get_open_ai_client()


example_input: dict[str, Any] = config.get("app").get("recommendation_example")

messages = example_input["messages"]

# Create a streaming request with custom inputs properly placed in extra_body
response_stream = openai_client.chat.completions.create(
    model=endpoint_name,
    messages=messages,
    temperature=0.0,
    max_tokens=100,
    stream=True,  # Enable streaming
    extra_body=example_input["configurable"]
)

# Process the streaming response
print("Streaming response:")
collected_content = ""
for chunk in response_stream:
    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
        content = chunk.choices[0].delta.content
        if content is not None:  # Check for None explicitly
            collected_content += content
            print(content, end="", flush=True)  # Print chunks as they arrive

print("\n\nFull collected response:")
print(collected_content)
