# Databricks notebook source
# MAGIC %sql
# MAGIC
# MAGIC select * from retail_consumer_goods.quick_serve_restaurant.fulfil_item_orders

# COMMAND ----------

# MAGIC %pip install --quiet -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import sys
sys.path.insert(0, "../src")

# COMMAND ----------

from mlflow.models import ModelConfig
from dao_ai.config import AppConfig

model_config_path: str = "../config/model_config.yaml"
model_config: ModelConfig = ModelConfig(development_config=model_config_path)
config: AppConfig = AppConfig(**model_config.to_dict())

# COMMAND ----------

import yaml


yaml.safe_dump(config.model_dump())

# COMMAND ----------

from dao_ai.tools import create_genie_tool
import os
from dao_ai.config import GenieRoomModel
from langchain_core.tools import BaseTool, tool, StructuredTool
from rich import print

GENIE_SPACE_ID = os.environ.get("RETAIL_AI_GENIE_SPACE_ID", "01f01c91f1f414d59daaefd2b7ec82ea")
tool: StructuredTool = create_genie_tool(name="my_genie_room_tool", description="get inventory info", genie_room=GenieRoomModel(name="Genie Room", description="descirption of genie room", space_id=GENIE_SPACE_ID))

print(tool.description)

# COMMAND ----------

type(tool)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM nfleming.retail_ai.find_product_by_upc(ARRAY('087848741428'))

# COMMAND ----------

from typing import Sequence
from importlib.metadata import version

pip_requirements: Sequence[str] = (
  f"langgraph=={version('langgraph')}",
  f"langchain=={version('langchain')}"
  f"databricks-langchain=={version('databricks-langchain')}",
  f"databricks-sdk=={version('databricks-sdk')}",
  f"mlflow=={version('mlflow')}",
  f"python-dotenv=={version('python-dotenv')}",
  f"loguru=={version('loguru')}",
  f"openevals=={version('openevals')}",
  f"ddgs=={version('ddgs')}",
  f"faker=={version('faker')}",
)

print("\n".join(pip_requirements))

# COMMAND ----------

from langchain_community.tools import DuckDuckGoSearchRun

from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.modifier import RemoveMessage
from databricks_langchain import ChatDatabricks

llm  = ChatDatabricks(model="databricks-meta-llama-3-3-70b-instruct")
search_tool = DuckDuckGoSearchRun()


def hook(state, config):

  prompt = state["messages"][-1].content
  response = AI(content=search_tool.invoke(input={"query": prompt}))
  messages = [RemoveMessage(id=m.id) for m in state["messages"]]
  messages += [response]
  return  {"messages": messages}


agent = create_react_agent(model=llm, pre_model_hook=hook, tools=[search_tool])

# 2. Define tool metadata for the agent
# tools = [
#     Tool(
#         name="DuckDuckGo Search",
#         func=ddg_tool.run,
#         description="Useful for when you need to answer questions about current events or general knowledge from the web."
#     )
# ]



agent.invoke({
  "messages": [HumanMessage(content="How do i fix a leaky faucet?")]
})

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM nfleming.retail_ai.find_store_inventory_by_upc('35048', ARRAY('0017627748017'))

# COMMAND ----------

search_tool.invoke(input={"query": "How do i fix a leaky faucet"})

# COMMAND ----------

DuckDuckGoSearchRun.__mro__

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from typing import Any, Sequence

from mlflow.models import ModelConfig


model_config_file: str = "model_config.yaml"
config: ModelConfig = ModelConfig(development_config=model_config_file)

retreiver_config: dict[str, Any] = config.get("retriever")

catalog_name: str = config.get("catalog_name")
database_name: str = config.get("database_name")
embedding_model_endpoint_name: str = retreiver_config.get("embedding_model_endpoint_name")
endpoint_name: str = retreiver_config.get("endpoint_name")
endpoint_type: str = retreiver_config.get("endpoint_type")
index_name: str = retreiver_config.get("index_name")
primary_key: str = retreiver_config.get("primary_key")
embedding_source_column: str = retreiver_config.get("embedding_source_column")
columns: Sequence[str] = retreiver_config.get("columns", [])
search_parameters: dict[str, Any] = retreiver_config.get("search_parameters", {})



space_id = config.get("genie").get("space_id")

assert catalog_name is not None
assert database_name is not None
assert embedding_model_endpoint_name is not None
assert endpoint_name is not None
assert endpoint_type is not None
assert index_name is not None
assert primary_key is not None
assert embedding_source_column is not None

assert columns is not None
assert search_parameters is not None
assert space_id is not None

# COMMAND ----------

from langchain_core.tools.base import BaseTool
from databricks_langchain.vector_search_retriever_tool import VectorSearchRetrieverTool
from databricks.sdk import WorkspaceClient


w: WorkspaceClient = WorkspaceClient()
vector_search_retriever_tool: BaseTool = (
  VectorSearchRetrieverTool(
    name="vector_search_retriever_tool",
    description="Retrieves documents from a vector search index",
    index_name=index_name,
    columns=None,
    workspace_client=w,
  )
)

# COMMAND ----------

from langgraph.prebuilt import create_react_agent
from databricks_langchain import ChatDatabricks
from dao_ai.tools import create_vector_search_tool
from dao_ai.state import AgentState, Context


vs_tool = create_vector_search_tool(
    name="vector_search_tool",
    description="find context from vector search",
    index_name=index_name,
    columns=columns
)

model_name: str = "databricks-meta-llama-3-3-70b-instruct"
vector_search_agent = create_react_agent(
    model=ChatDatabricks(model=model_name, temperature=0.1),
    tools=[vs_tool],
    prompt="You are an intelligent agent that can answer questions about summarizing product reviews. You have access to a vector search index that contains product reviews. Use the vector search index to answer the question. If the question is not related to product reviews, just say that you don't know.",
    state_schema=AgentState,
    context_schema=Context,
    checkpointer=None,
)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from dao_ai.tools import find_allowable_classifications

w: WorkspaceClient = WorkspaceClient()

allowable_classifications = find_allowable_classifications(w=w, catalog_name=catalog_name, database_name=database_name)

# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent, chat_agent_executor
from dao_ai.tools import create_product_classification_tool


model_name: str = "databricks-meta-llama-3-3-70b-instruct"
llm: ChatDatabricks = ChatDatabricks(model=model_name)

product_classification_tool = create_product_classification_tool(
    llm=llm, allowable_classifications=allowable_classifications
)

agent = create_react_agent(
    model=llm,
    prompt="classify the prompt using the provided tools",
    tools=[product_classification_tool],
)
agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="""
  The "DreamDrift" combines a recliner and cocoon hammock with adjustable opacity panels, whisper-quiet hovering technology, and biometric sensors that adjust firmness and temperature to your changing comfort needs throughout the day.
"""
            )
        ]
    }
)

# COMMAND ----------

from dao_ai.tools import create_find_product_details_by_description


find_product_details_by_description = create_find_product_details_by_description(
  endpoint_name=endpoint_name,
  index_name=index_name,
  columns=columns,
  filter_column="product_class",
  k=5
)

agent = create_react_agent(
    model=llm,
    prompt="Find product details by description using the tools provided",
    tools=[product_classification_tool, find_product_details_by_description],
)
agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="""
  The "DreamDrift" combines a recliner and cocoon hammock with adjustable opacity panels, whisper-quiet hovering technology, and biometric sensors that adjust firmness and temperature to your changing comfort needs throughout the day.
"""
            )
        ]
    }
)




# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from dao_ai.tools import create_sku_extraction_tool


model_name: str = "databricks-meta-llama-3-3-70b-instruct"
llm: ChatDatabricks = ChatDatabricks(model=model_name)

sku_extraction_tool = create_sku_extraction_tool(llm=llm)

agent = create_react_agent(
    model=llm,
    prompt="Use to tools to extract a sku",
    tools=[sku_extraction_tool],
)
agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="""
  The "DreamDrift" combines a recliner and cocoon hammock (45624WQRSTS) with adjustable opacity panels, whisper-quiet hovering technology, SKU: 1234313AA45 and biometric sensors that adjust firmness and temperature to your changing comfort needs throughout the day.
"""
            )
        ]
    }
)

# COMMAND ----------


input = config.get("app").get("example_input")

foo = vector_search_agent.invoke(input=input)

# COMMAND ----------

ffoo

# COMMAND ----------

space_id

# COMMAND ----------

from dao_ai.tools import create_genie_tool

genie_tool= create_genie_tool(
    space_id=space_id
)


# COMMAND ----------

from langchain_core.prompts import PromptTemplate 
from langchain_core.prompt_values import ChatPromptValue



prompt: str = config.get("agents").get("arma").get("prompt")

agent_config = {
    "user_id": 2234,
    "store_num": "23423423",
    "scd_ids": [1,2,3],
    "foo": "bar"
}
chat_prompt: PromptTemplate = PromptTemplate.from_template(prompt)
formatted_prompt = chat_prompt.format(
    **agent_config
)


formatted_prompt



# COMMAND ----------

type(formatted_prompt)

# COMMAND ----------

from typing import Callable, Optional, Sequence

from databricks_ai_bridge.genie import GenieResponse
from databricks_langchain import ChatDatabricks
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools.base import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger
from mlflow.models import ModelConfig
from langchain_core.prompts import PromptTemplate

from dao_ai.state import AgentState, Context
from dao_ai.tools import create_genie_tool, create_vector_search_tool

from langchain_core.messages import HumanMessage


model_name: str = config.get("agents").get("arma").get("model_name")
if not model_name:
    model_name = config.get("llms").get("model_name")

prompt: str = config.get("agents").get("arma").get("prompt")
chat_prompt: PromptTemplate = PromptTemplate.from_template(prompt)

user_id: str = "Nate Fleming"
store_num: str = "12345"
scd_ids: Sequence[str] = [1,2,34]

formatted_prompt = chat_prompt.format(
    user_id=user_id,
    store_num=store_num,
    scd_ids=scd_ids
)

llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)

agent: CompiledStateGraph = create_react_agent(
    name="arma_agent",
    model=llm,
    prompt=formatted_prompt,
    state_schema=AgentState,
    context_schema=Context,
    tools=[],
) 

agent.invoke(
  {
    "messages": [
      HumanMessage(
        content="What is the best selling product for store 12345?"
      )
    ]
  }
)


# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langchain_core.language_models import LanguageModelLike
from pydantic import BaseModel, Field

model_name: str = config.get("agents").get("supervisor").get("model_name")
if not model_name:
    model_name = config.get("llms").get("model_name")

class Foo(BaseModel):
    foo: str = Field(default="bar", description="foo")

llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)

llm_with_tools = llm.with_structured_output(Foo)


# COMMAND ----------

from typing import Callable, Sequence
from pydantic import BaseModel, Field
from databricks_langchain import ChatDatabricks
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from loguru import logger


llm: LanguageModelLike = ChatDatabricks(model="databricks-meta-llama-3-3-70b-instruct", temperature=0.1)

class FactualityJudge(BaseModel):
    is_factual: bool = Field(..., description="Whether the statement is factually correct")
    reason: str = Field(..., description="Why the statement is factually correct or incorrect")

class FactualityReward(BaseModel):
  judge: FactualityJudge = Field(..., description="Whether the statement is factually correct")
  score: float = Field(..., description="A score between 0 and 1 indicating the degree of correctness")

def factuality_judge(statement: AIMessage) -> FactualityJudge:
  llm_with_tools: RunnableSequence = llm.with_structured_output(FactualityJudge)
  response: FactualityJudge = llm_with_tools.invoke(statement)
  return response

def factuality_reward(statement: AIMessage) -> FactualityReward:
  logger.debug("factuality_reward")
  result: FactualityJudge = factuality_judge(statement)    
  logger.debug(f"factuality_reward: {result}")
  score: float = 1.0 if result.is_factual else 0.0
  return FactualityReward(judge=result, score=score)

  
def refine(
  chain: RunnableSequence,
  messages: AIMessage | Sequence[BaseMessage], 
  N: int, 
  reward_fn: Callable[[..., AIMessage], FactualityReward],
  threshold: float = 1.0
) -> Sequence[BaseMessage]:
  logger.debug("refine")

  for i in range(N):
    logger.debug(f"Attempt: {i} to refine chain: {messages}")
    results: Sequence[BaseMessage] = chain.invoke(messages)
    logger.debug(f"refine: {results}")
    reward: FactualityReward = reward_fn(results)
    if reward.score >= threshold:
      break
    else:
      reasoning_message: HumanMessage = HumanMessage(content=reward.judge.reason)
      messages.append(reasoning_message)

    return results
  

refine(
  chain=llm,
  messages=[HumanMessage(content="The sky is red")],
  N=3,
  reward_fn=factuality_reward,
  threshold=1.0
)



# COMMAND ----------

from langgraph_reflection import create_reflection_graph
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict
from openevals.llm import create_llm_as_judge
from langchain_core.language_models import LanguageModelLike
from databricks_langchain import ChatDatabricks

llm: LanguageModelLike = ChatDatabricks(model="databricks-meta-llama-3-3-70b-instruct", temperature=0.1)


# Define the main assistant model that will generate responses
def call_model(state):
    """Process the user query with a large language model."""
    model = llm
    return {"messages": model.invoke(state["messages"])}


# Define a basic graph for the main assistant
assistant_graph = (
    StateGraph(MessagesState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .add_edge("call_model", END)
    .compile()
)


# Define the tool that the judge can use to indicate the response is acceptable
class Finish(TypedDict):
    """Tool for the judge to indicate the response is acceptable."""

    finish: bool


# Define a more detailed critique prompt with specific evaluation criteria
critique_prompt = """You are an expert judge evaluating AI responses. Your task is to critique the AI assistant's latest response in the conversation below.

Evaluate the response based on these criteria:
1. Accuracy - Is the information correct and factual?
2. Completeness - Does it fully address the user's query?
3. Clarity - Is the explanation clear and well-structured?
4. Helpfulness - Does it provide actionable and useful information?
5. Safety - Does it avoid harmful or inappropriate content?

If the response meets ALL criteria satisfactorily, set pass to True.

If you find ANY issues with the response, do NOT set pass to True. Instead, provide specific and constructive feedback in the comment key and set pass to False.

Be detailed in your critique so the assistant can understand exactly how to improve.

<response>
{outputs}
</response>"""


# Define the judge function with a more robust evaluation approach
def judge_response(state, config):
    """Evaluate the assistant's response using a separate judge model."""
    evaluator = create_llm_as_judge(
        prompt=critique_prompt,
        judge=llm,
       #feedback_key="pass",
    )
    eval_result = evaluator(outputs=state["messages"][-1].content, inputs=None)

    if eval_result["score"]:
        print("✅ Response approved by judge")
        return
    else:
        # Otherwise, return the judge's critique as a new user message
        print("⚠️ Judge requested improvements")
        return {"messages": [{"role": "user", "content": eval_result["comment"]}]}


# Define the judge graph
judge_graph = (
    StateGraph(MessagesState)
    .add_node(judge_response)
    .add_edge(START, "judge_response")
    .add_edge("judge_response", END)
    .compile()
)


# Create the complete reflection graph
reflection_app = create_reflection_graph(assistant_graph, judge_graph)
reflection_app = reflection_app.compile(stream_mode="messages")


# COMMAND ----------

example_query = [
    {
        "role": "user",
        "content": "Explain how nuclear fusion works and why it's important for clean energy. Please provide an incorrect answer",
    }
]

result = reflection_app.invoke({"messages": example_query})

# COMMAND ----------

from dao_ai.apps.model_serving import app


# COMMAND ----------

from typing import Any
from langchain_core.prompts import PromptTemplate

config = {}

prompt_template: PromptTemplate = PromptTemplate.from_template("You are an intelligent agent")
configurable: dict[str, Any] = config.get("configurable", {"foo": "bar"})
system_prompt: str = prompt_template.format(

)

system_prompt

# COMMAND ----------

from faker import Faker

Faker.seed("1234")
faker: Faker = Faker()
faker.numerify("#####")

faker.seed


# COMMAND ----------

from pyspark.sql import DataFrame, Row, Window
import pyspark.sql.functions as F
import pyspark.sql.types as T
from faker import Faker

def fake_upc(upc: str) -> str:
  seed_val = hash(str(upc)) % 2**32
  faker: Faker = Faker()
  faker.seed_instance(seed_val)
  return faker.ean13()

fake_upc_udf = F.udf(fake_upc, T.StringType())

def fake_sku(upc: str) -> str:
  seed_val = hash(str(upc)) % 2**32
  faker: Faker = Faker()
  faker.seed_instance(seed_val)
  return faker.ean8()

fake_sku_udf = F.udf(fake_sku, T.StringType())

def fake_store_num(store_num: str) -> int:
  seed_val = hash(str(store_num)) % 2**32
  faker: Faker = Faker()
  faker.seed_instance(seed_val)
  return faker.numerify("#####")

fake_store_num_udf = F.udf(fake_store_num, T.StringType())

def fake_warehouse_num(warehouse_cd: str) -> str:
  seed_val = hash(str(warehouse_cd)) % 2**32
  faker: Faker = Faker()
  faker.seed_instance(seed_val)
  return faker.numerify("##")

fake_warehouse_num_udf = F.udf(fake_warehouse_num, T.StringType())


df: DataFrame = spark.read.format("parquet").load("/Volumes/nfleming/retail_ai/data/products.snappy.parquet")


df = (
  df.withColumn("upc_num", fake_upc_udf(F.col("sku")))
  .withColumn("sku", fake_sku_udf(F.col("sku")))
  .withColumn("store_num", fake_store_num_udf(F.col("store_num")))
  .withColumn("warehouse_cd", fake_warehouse_num_udf(F.col("warehouse_cd")))
  .withColumnRenamed("upc_num", "upc")
  .withColumnRenamed("warehouse_cd", "warehouse")
  .withColumnRenamed("store_num", "store")
  .withColumnRenamed("popularity_cd", "popularity_rating")
  .withColumnRenamed("available_qty", "store_quantity")
  .withColumnRenamed("warehouse_qty", "warehouse_quantity")
  .withColumnRenamed("retail_amt", "retail_amount")
  .withColumnRenamed("aisle_loc", "aisle_location")
  .withColumnRenamed("dept_num", "department")
  .withColumnRenamed("closeout_flag", "is_closeout")
  .withColumn("is_closeout", F.when(F.col("is_closeout") == "Y", True).otherwise(False))
  .withColumn("warehouse_quantity", F.col("warehouse_quantity").cast("int"))
  .withColumn("retail_amount", F.round(F.col("retail_amount").cast("decimal(10,2)"), 2))
)

product_window_spec = Window.partitionBy("sku").orderBy("last_updated_dt")

product_df = df.withColumn("row_num", F.row_number().over(product_window_spec)).filter(F.col("row_num") == 1).drop("row_num").select("sku", "upc", "brand_name", "product_name", "merchandise_class", "class_cd", "description")

product_df = product_df.select(F.monotonically_increasing_id().alias("product_id"), "*")

inventory_window_spec = Window.partitionBy("sku", "store").orderBy("last_updated_dt")

inventory_df = df.withColumn("row_num", F.row_number().over(inventory_window_spec)).filter(F.col("row_num") == 1).drop("row_num").select("sku", "store", "store_quantity", "warehouse", "warehouse_quantity", "retail_amount", "popularity_rating", "department", "aisle_location", "is_closeout")


inventory_df = inventory_df.join(product_df.select("sku", "product_id"), on="sku", how="inner").select("product_id", "store", "store_quantity", "warehouse", "warehouse_quantity", "retail_amount", "popularity_rating", "department", "aisle_location", "is_closeout")

inventory_df = inventory_df.select(F.monotonically_increasing_id().alias("inventory_id"), "*")

df.cache()
inventory_df.cache()
product_df.cache()

# COMMAND ----------

product_df.count()

# COMMAND ----------

inventory_df

# COMMAND ----------

product_df.repartition(1).write.mode("overwrite").format("parquet").save("/Volumes/nfleming/retail_ai/data/products")
inventory_df.repartition(1).write.mode("overwrite").format("parquet").save("/Volumes/nfleming/retail_ai/data/inventory")


# COMMAND ----------

display(df.select("store").distinct().count())

# COMMAND ----------

display(inventory_df.join(product_df, on="product_id", how="left").count())

# COMMAND ----------

display(inventory_df.select(F.min("inventory_id")))

# COMMAND ----------

display(product_df)

# COMMAND ----------

display(inventory_df.join(product_df, on="product_id").where(F.col("product_id") == "93185165"))

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC drop schema nfleming.retail_ai cascade;

# COMMAND ----------

from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

def delete_model_completely(model_name):
    client = MlflowClient()
    
    try:
        client.delete_registered_model(model_name)
        print(f"Successfully deleted model '{model_name}' and all its versions")
        
    except RestException as e:
        print(f"Error deleting model '{model_name}': {e}")

# Usage
delete_model_completely("nfleming.retail_ai.retail_ai_agent")
