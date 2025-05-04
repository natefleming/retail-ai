# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "langgraph",
  "langchain",
  "databricks-langchain",
  "unitycatalog-langchain[databricks]",
  "langgraph-checkpoint-postgres",
  "duckduckgo-search",
  "databricks-agents",
  "psycopg[binary,pool]", 
  "databricks-sdk",
  "langgraph-reflection",
  "openevals",
  "mlflow",
  "pydantic",
  "python-dotenv",
  "uv",
  "grandalf",
  "loguru",
  "rich"
)

pip_requirements: str = " ".join(pip_requirements)

%pip install --quiet --upgrade {pip_requirements}
%restart_python

# COMMAND ----------

from typing import Sequence
from importlib.metadata import version

pip_requirements: Sequence[str] = (
    f"langgraph=={version('langgraph')}",
    f"langchain=={version('langchain')}",
    f"databricks-langchain=={version('databricks-langchain')}",
    f"unitycatalog-langchain[databricks]=={version('unitycatalog-langchain')}",
    f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
    f"duckduckgo-search=={version('duckduckgo-search')}",
    f"databricks-sdk=={version('databricks-sdk')}",
    f"langgraph-reflection=={version('langgraph-reflection')}",
    f"openevals=={version('openevals')}",
    f"mlflow=={version('mlflow')}",
    f"psycopg[binary,pool]=={version('psycopg')}",
    f"databricks-agents=={version('databricks-agents')}",
    f"pydantic=={version('pydantic')}",
    f"loguru=={version('loguru')}",
)

print("\n".join(pip_requirements))

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recommendation

# COMMAND ----------

from typing import Any
from rich import print as pprint
from agent_as_code import app, config
from retail_ai.models import process_messages

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("recommendation_example")
pprint(input_example)

response = process_messages(app=app, input=input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from agent_as_code import app, config
from retail_ai.models import process_messages_stream

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("recommendation_example")
pprint(input_example)

for event in process_messages_stream(app=app, input=input_example):
  print(event.choices[0].delta.content, end="", flush=True)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Inventory

# COMMAND ----------

from typing import Any
from rich import print as pprint
from agent_as_code import app, config
from retail_ai.models import process_messages

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("inventory_example")
pprint(input_example)

response = process_messages(app=app, input=input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from agent_as_code import app, config
from retail_ai.models import process_messages_stream

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("inventory_example")
pprint(input_example)

for event in process_messages_stream(app=app, input=input_example):
  print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparison

# COMMAND ----------

from typing import Any
from rich import print as pprint
from agent_as_code import app, config
from retail_ai.models import process_messages

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("comparison_example")
pprint(input_example)

response = process_messages(app=app, input=input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from agent_as_code import app, config
from retail_ai.models import process_messages_stream

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("comparison_example")
pprint(input_example)

for event in process_messages_stream(app=app, input=input_example):
  print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## General

# COMMAND ----------

from typing import Any
from rich import print as pprint
from agent_as_code import app, config
from retail_ai.models import process_messages

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("general_example")
pprint(input_example)

response = process_messages(app=app, input=input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from agent_as_code import app, config
from retail_ai.models import process_messages_stream

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("general_example")
pprint(input_example)

for event in process_messages_stream(app=app, input=input_example):
  print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DIY

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from typing import Any
from rich import print as pprint
from agent_as_code import app, config
from retail_ai.models import process_messages

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("diy_example")
pprint(input_example)

response = process_messages(app=app, input=input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from agent_as_code import app, config
from retail_ai.models import process_messages_stream

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("diy_example")
pprint(input_example)

for event in process_messages_stream(app=app, input=input_example):
  print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Orders

# COMMAND ----------

from typing import Any
from rich import print as pprint
from agent_as_code import app, config
from retail_ai.models import process_messages

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("orders_example")
pprint(input_example)

response = process_messages(app=app, input=input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from agent_as_code import app, config
from retail_ai.models import process_messages_stream

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("orders_example")
pprint(input_example)

for event in process_messages_stream(app=app, input=input_example):
  print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Product

# COMMAND ----------

from typing import Any
from rich import print as pprint
from agent_as_code import app, config
from retail_ai.models import process_messages

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("product_example")
pprint(input_example)

response = process_messages(app=app, input=input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from agent_as_code import app, config
from retail_ai.models import process_messages_stream

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("product_example")
pprint(input_example)

for event in process_messages_stream(app=app, input=input_example):
  print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from typing import Any, Sequence
from rich import print as pprint

from langchain_core.messages import HumanMessage
from agent_as_code import app, config
from retail_ai.models import _process_langchain_messages_stream

examples: dict[str, Any] = config.get("app").get("examples")
input_example: dict[str, Any] = examples.get("product_image_example")
pprint(input_example)

messages: Sequence[HumanMessage] = [
  HumanMessage(content=m["content"]) for m in input_example["messages"]
]
config: dict[str, Any] = input_example["custom_inputs"]
messages
for event in _process_langchain_messages_stream(app=app, messages=messages, config=config):
  print(event.content, end="", flush=True)

# COMMAND ----------

type(response)
