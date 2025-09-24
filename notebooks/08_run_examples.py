# Databricks notebook source
# MAGIC %pip install --quiet --upgrade -r ../requirements.txt
# MAGIC %pip uninstall --quiet -y databricks-connect pyspark pyspark-connect
# MAGIC %pip install --quiet databricks-connect
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

import sys

sys.path.insert(0, "../src")

# COMMAND ----------

import dao_ai.providers
import dao_ai.providers.base
import dao_ai.providers.databricks

# COMMAND ----------

import nest_asyncio
nest_asyncio.apply()

# COMMAND ----------

from typing import Sequence
import os

def find_yaml_files_os_walk(base_path: str) -> Sequence[str]:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path does not exist: {base_path}")
    
    if not os.path.isdir(base_path):
        raise NotADirectoryError(f"Base path is not a directory: {base_path}")
    
    yaml_files = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.yaml', '.yml')):
                yaml_files.append(os.path.join(root, file))
    
    return sorted(yaml_files)

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")

config_files: Sequence[str] = find_yaml_files_os_walk("../config")
dbutils.widgets.dropdown(name="config-paths", choices=config_files, defaultValue=next(iter(config_files), ""))

config_path: str | None = dbutils.widgets.get("config-path") or None
project_path: str = dbutils.widgets.get("config-paths") or None

config_path: str = config_path or project_path

print(config_path)

# COMMAND ----------

import sys
import mlflow
from mlflow.pyfunc import ChatModel
from dao_ai.config import AppConfig

from loguru import logger

mlflow.langchain.autolog()

config: AppConfig = AppConfig.from_file(path=config_path)

app: ChatModel = config.as_chat_model()

# COMMAND ----------

config.display_graph()

# COMMAND ----------

from typing import Any, Sequence
import yaml
from pathlib import Path
from rich import print as pprint


examples_path: Path = Path.cwd().parent / "examples"
projects: Sequence[str] = [item.name for item in examples_path.iterdir() if item.is_dir()]

dbutils.widgets.dropdown(name="example-project", defaultValue=projects[0], choices=projects)
project: str = dbutils.widgets.get("example-project")

project_examples: Path = Path.cwd().parent / "examples" / project
examples_files: Sequence[str] = [item.name for item in project_examples.iterdir() if item.is_file()]

dbutils.widgets.dropdown(name="example_files", defaultValue=examples_files[0], choices=examples_files)
example_file: str = dbutils.widgets.get("example_files")

chosen_example: str | None = None
chosen_input_example: dict[str, Any] = {}
examples_path: Path = Path.cwd().parent / "examples" / project / example_file
if examples_path.exists():
  retail_examples: dict[str, Any] = yaml.safe_load(examples_path.read_text())

  examples: dict[str, Any] = retail_examples.get("examples", {})

  example_names: Sequence[str] = sorted(examples.keys())

  dbutils.widgets.dropdown(name="example", defaultValue=example_names[0], choices=example_names)
  chosen_example: dict[str, Any] = dbutils.widgets.get("example")

  chosen_input_example = examples.get(chosen_example, {})

pprint(chosen_example)
pprint(chosen_input_example)




# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages

pprint(chosen_input_example)

response = process_messages(app=app, **chosen_input_example)
pprint(response)

# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages_stream

pprint(chosen_input_example)

for event in process_messages_stream(app=app, **chosen_input_example):
  print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

from typing import Any
from rich import print as pprint
from dao_ai.models import process_messages

# store num
input_example: dict[str, Any] = {
  'messages': [
    {
      'role': 'user',
      'content': 'Can I have a medium latte?'
    }
  ],
  'custom_inputs': {
      'configurable': {
        'thread_id': '1',
        'user_id': 'nate.fleming',
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


