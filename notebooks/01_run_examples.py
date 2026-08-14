# Databricks notebook source
# Dependency bootstrap, matching the pipeline step notebooks. Install dao-ai —
# which pulls its own transitive deps, python-dotenv and nest-asyncio included —
# from the newest ../dist wheel if one is there, else the published PyPI package.
# This notebook builds agent graphs out of arbitrary examples, so it needs every
# optional feature extra: ``[all]``. The spec is single-quoted in the magic so a
# dev wheel's ``+local`` version tag and the bracket survive shell expansion.
#
# Note: with no ../dist wheel this installs the *published* dao-ai, not your
# working tree. Run ``uv build`` from the repo root first to test local changes.
import glob, os

from packaging.version import Version

# Newest by *version*, not by filename: a lexical sort puts 0.2.8 above
# 0.2.10. ``Version`` also parses a dev wheel's ``+local`` tag correctly.
def _wheel_version(wheel: str) -> Version:
    return Version(os.path.basename(wheel).split("-")[1])

_wheels = sorted(glob.glob("../dist/dao_ai-*.whl"), key=_wheel_version, reverse=True)
_dao_ai_dep = (_wheels[0] if _wheels else "dao-ai") + "[all]"

# MAGIC %uv pip install --quiet '{_dao_ai_dep}'
# MAGIC %pip uninstall --quiet -y pyspark pyspark-connect
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

import sys, os, glob, subprocess

from packaging.version import Version

# Newest by *version*, not by filename: a lexical sort puts 0.2.8 above
# 0.2.10. ``Version`` also parses a dev wheel's ``+local`` tag correctly.
def _wheel_version(wheel: str) -> Version:
    return Version(os.path.basename(wheel).split("-")[1])

_wheels = sorted(
    glob.glob("../dist/dao_ai-*.whl") or glob.glob("../../artifacts/.internal/dao_ai-*.whl"),
    key=_wheel_version,
    reverse=True,
)
if _wheels:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall", _wheels[0]])
elif os.path.isdir("../src/dao_ai"):
    sys.path.insert(0, "../src")

# COMMAND ----------

import dao_ai.providers
import dao_ai.providers.base
import dao_ai.providers.databricks

# COMMAND ----------

import nest_asyncio
nest_asyncio.apply()

# COMMAND ----------

# There is no `../config` discovery fallback. That directory does not exist in a
# repo checkout, and discovery would pick the first YAML it happened to find, so
# `config-path` is the single input. It defaults to a shipped example; point it at
# any other config under ../examples.
dbutils.widgets.text(
    name="config-path",
    defaultValue="../examples/99_complete_applications/hardware_store/hardware_store.yaml",
)

widget_path: str | None = dbutils.widgets.get("config-path") or None
if not widget_path:
    raise ValueError(
        "No config to run: the `config-path` widget is empty. Set it to a config "
        "YAML — relative to this notebook (e.g. "
        "`../examples/99_complete_applications/hardware_store/hardware_store.yaml`) "
        "or an absolute workspace path."
    )

config_path: str = widget_path

print(config_path)

# COMMAND ----------

import sys
import mlflow
from mlflow.pyfunc import ChatModel
from dao_ai.config import AppConfig

from loguru import logger

mlflow.langchain.autolog(run_tracer_inline=True)

config: AppConfig = AppConfig.from_file(path=config_path)

app: ChatModel = config.as_chat_model()

# COMMAND ----------

config.display_graph()

# COMMAND ----------

from typing import Any, Sequence
import yaml
from pathlib import Path
from rich import print as pprint


# Inference examples are colocated with each complete-application config as
# ``<use-case>/examples.yaml``. Discover the use-case dirs that ship one.
apps_root: Path = (
    Path.cwd().parent / "examples" / "99_complete_applications"
)
projects: Sequence[str] = (
    sorted(
        item.name
        for item in apps_root.iterdir()
        if item.is_dir() and (item / "examples.yaml").exists()
    )
    if apps_root.is_dir()
    else []
)
if not projects:
    # The same trap the `config-path` widget above used to have: `iterdir()`
    # raises `FileNotFoundError` on a missing directory and `dropdown` rejects an
    # empty `choices` list, so both cases have to be caught *before* the widget.
    raise ValueError(
        f"No inference examples found under {apps_root}: this cell needs the "
        "repo's `examples/` tree as a sibling of `notebooks/`. Upload the repo "
        "rather than this notebook alone, or stop here — the agent built above "
        "is already usable, just call it with your own input."
    )

dbutils.widgets.dropdown(name="example-project", defaultValue=projects[0], choices=projects)
project: str = dbutils.widgets.get("example-project")

chosen_example: str | None = None
chosen_input_example: dict[str, Any] = {}
examples_path: Path = apps_root / project / "examples.yaml"
if examples_path.exists():
  retail_examples: dict[str, Any] = yaml.safe_load(examples_path.read_text())

  examples: dict[str, Any] = retail_examples.get("examples", {})

  example_names: Sequence[str] = sorted(examples.keys())

  # An `examples.yaml` with no `examples:` block would hit the same empty-choices
  # rejection; leave the widget uncreated and fall through to the empty defaults.
  if example_names:
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


