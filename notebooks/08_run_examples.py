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

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="../config/model_config.yaml")
config_path: str = dbutils.widgets.get("config-path")
print(config_path)

# COMMAND ----------

from typing import Sequence
from importlib.metadata import version
import sys


sys.path.insert(0, "../src")

pip_requirements: Sequence[str] = (
    f"langgraph=={version('langgraph')}",
    f"langchain=={version('langchain')}",
    f"databricks-langchain=={version('databricks-langchain')}",
    f"unitycatalog-langchain[databricks]=={version('unitycatalog-langchain')}",
    f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
    f"duckduckgo-search=={version('duckduckgo-search')}",
    f"databricks-sdk=={version('databricks-sdk')}",
    f"openevals=={version('openevals')}",
    f"mlflow=={version('mlflow')}",
    f"psycopg[binary,pool]=={version('psycopg')}",
    f"databricks-agents=={version('databricks-agents')}",
    f"pydantic=={version('pydantic')}",
    f"loguru=={version('loguru')}",
)

print("\n".join(pip_requirements))

# COMMAND ----------

import nest_asyncio
nest_asyncio.apply()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from typing import Any
import yaml
from pathlib import Path

retail_examples_path: Path = Path.cwd().parent / "examples" / "retail_examples.yaml"
retail_examples: dict[str, Any] = yaml.safe_load(retail_examples_path.read_text())

# COMMAND ----------

import sys
import mlflow
from langgraph.graph.state import CompiledStateGraph
from mlflow.pyfunc import ChatModel
from dao_ai.graph import create_dao_ai_graph
from dao_ai.models import create_agent 
from dao_ai.config import AppConfig

from loguru import logger

mlflow.langchain.autolog()

config: AppConfig = AppConfig.from_file(path=config_path)

log_level: str = config.app.log_level

logger.add(sys.stderr, level=log_level)

graph: CompiledStateGraph = create_dao_ai_graph(config=config)

app: ChatModel = create_agent(graph)


# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages

# store num
input_example: dict[str, Any] = {
  'messages': [
    {
      'role': 'user',
      'content': 'How many big green egg grills do you have in stock?'
    }
  ],
  'custom_inputs': {
      'configurable': {
        'thread_id': '1',
        'user_id': 'nate.fleming',
        'store_num': 35048
      }
    }
  }
pprint(input_example)

response = process_messages(app=app, **input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages

# store num
input_example: dict[str, Any] = {
  'messages': [
    {
      'role': 'user',
      'content': 'How many of 0017627748017 do you have in stock in my store?'
    }
  ],
  'custom_inputs': {
      'configurable': {
        'thread_id': '1',
        'user_id': 'nate.fleming',
        'store_num': 35048
      }
    }
  }
pprint(input_example)

response = process_messages(app=app, **input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages

# store num
input_example: dict[str, Any] = {
  'messages': [
    {
      'role': 'user',
      'content': 'Can you tell me about 0017627748017?'
    }
  ],
  'custom_inputs': {
      'configurable': {
        'thread_id': '1',
        'user_id': 'nate.fleming',
        'store_num': 123
      }
    }
  }
pprint(input_example)

response = process_messages(app=app, **input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages

# store num
input_example: dict[str, Any] = {
  'messages': [
    {
      'role': 'user',
      'content': 'Can you tell me about sku 00176279?'
    }
  ],
  'custom_inputs': {
      'configurable': {
        'thread_id': '1',
        'user_id': 'nate.fleming',
        'store_num': 123
      }
    }
  }
pprint(input_example)

response = process_messages(app=app, **input_example)
pprint(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recommendation

# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("recommendation_example")

response = process_messages(app=app, **input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from dao_ai.models import process_messages_stream

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("recommendation_example")
pprint(input_example)

for event in process_messages_stream(app=app, **input_example):
  print(event.choices[0].delta.content, end="", flush=True)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Inventory

# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("inventory_example")
pprint(input_example)

response = process_messages(app=app, **input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from dao_ai.models import process_messages_stream

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("inventory_example")
pprint(input_example)

for event in process_messages_stream(app=app, **input_example):
  print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparison

# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("comparison_example")
pprint(input_example)

response = process_messages(app=app, **input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from dao_ai.models import process_messages_stream

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("comparison_example")
pprint(input_example)

for event in process_messages_stream(app=app, **input_example):
  print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

from typing import Any, Sequence
from rich import print as pprint

from pathlib import Path
from langchain_core.messages import HumanMessage, convert_to_messages
from dao_ai.models import process_messages
from dao_ai.messages import convert_to_langchain_messages


examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("comparison_image_example")
pprint(input_example)

messages: Sequence[HumanMessage] = convert_to_langchain_messages(input_example["messages"])
custom_inputs = input_example["custom_inputs"]

process_messages(
  app=app, 
  messages=messages, 
  custom_inputs=custom_inputs
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## General

# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("comparison_image_example")
pprint(input_example)

response = process_messages(app=app, **input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from dao_ai.models import process_messages_stream

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("general_example")
pprint(input_example)

for event in process_messages_stream(app=app, **input_example):
  print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DIY

# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("diy_example")
pprint(input_example)

response = process_messages(app=app, **input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from dao_ai.models import process_messages_stream

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("diy_example")
pprint(input_example)

for event in process_messages_stream(app=app, **input_example):
  print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Orders

# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("orders_example")
pprint(input_example)

response = process_messages(app=app, **input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from dao_ai.models import process_messages_stream

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("orders_example")
pprint(input_example)

for event in process_messages_stream(app=app, **input_example):
  print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Product

# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("product_example")
pprint(input_example)

response = process_messages(app=app, **input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages_stream

examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("product_example")
pprint(input_example)

for event in process_messages_stream(app=app, **input_example):
  print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

from typing import Any, Sequence
from rich import print as pprint

from pathlib import Path
from langchain_core.messages import HumanMessage, convert_to_messages
from dao_ai.models import process_messages
from dao_ai.messages import convert_to_langchain_messages


examples: dict[str, Any] = retail_examples.get("examples")
input_example: dict[str, Any] = examples.get("product_image_example")
pprint(input_example)

messages: Sequence[HumanMessage] = convert_to_langchain_messages(input_example["messages"])
custom_inputs = input_example["custom_inputs"]

process_messages(
  app=app, 
  messages=messages, 
  custom_inputs=custom_inputs
)


