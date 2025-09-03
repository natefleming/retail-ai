# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Decarative Agent Orchestration Framework (DAO-AI)
# MAGIC ### [https://github.com/natefleming/dao-ai](https://github.com/natefleming/dao-ai)

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade dao-ai
# MAGIC %pip uninstall --quiet -y databricks-connect pyspark pyspark-connect
# MAGIC %pip install --quiet databricks-connect
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %pip install --quiet -r ../requirements.txt
# MAGIC %pip uninstall --quiet -y databricks-connect pyspark pyspark-connect
# MAGIC %pip install --quiet databricks-connect
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.dropdown("app", "genie", ["quick-serve-restaurant", "hardware-store", "genie"])

# COMMAND ----------

import sys
import nest_asyncio
nest_asyncio.apply()

sys.path.insert(0, "../src")


# COMMAND ----------

path_map: dict[str, str] = {
    "quick-serve-restaurant": "../config/quick_serve_restaurant/quick-serve-restaurant.yaml",
    "hardware-store": "../config/hardware_store/supervisor.yaml",
    "genie": "../config/examples/genie.yaml",
}

question_map: dict[str, str] = {
  "quick-serve-restaurant": "What are the ingredients in a Vanilla Latte?",
  "hardware-store": "How many big green egg grills do you have in stock?",
  "genie": "What are the first 5 products in the product table"
}
config_path: str = path_map.get(dbutils.widgets.get("app"))
question: str = question_map.get(dbutils.widgets.get("app"))
print(f"Using config file: {config_path}")
print(f"Using question: {question}")

# COMMAND ----------

from dao_ai.config import AppConfig

app: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

app.display_graph()

# COMMAND ----------

from dao_ai.models import process_messages_stream
from mlflow.pyfunc import ChatModel

chat_model: ChatModel = app.as_chat_model()

for event in process_messages_stream(
    app=chat_model,
    messages=[
        {"role": "user", "content": question},
    ],
    custom_inputs={
        "configurable": {
            "thread_id": "17df0e6b-17cc-4aeb-9d34-987ea9c8fd97",
            "user_id": "my_user_id",
            "store_num": 87887,
        }
    },
):
    print(event.choices[0].delta.content, end="", flush=True)

# COMMAND ----------

app.create_agent()

# COMMAND ----------

app.deploy_agent()
