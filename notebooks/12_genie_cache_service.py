# Databricks notebook source
# MAGIC %md
# MAGIC # Genie Cache Service Demo
# MAGIC
# MAGIC This notebook demonstrates how to use the Genie cache service with both:
# MAGIC - **PostgresContextAwareGenieService**: Persistent context-aware cache using PostgreSQL/Lakebase
# MAGIC - **LRUCacheService**: In-memory LRU cache for exact match lookups
# MAGIC
# MAGIC The cache layers are chained together using the decorator pattern:
# MAGIC ```
# MAGIC LRU Cache -> Context-Aware Cache -> Genie API
# MAGIC ```
# MAGIC
# MAGIC When a question is asked:
# MAGIC 1. LRU cache checks for exact match (fast, O(1))
# MAGIC 2. If miss, context-aware cache checks for semantic match (using embeddings)
# MAGIC 3. If miss, Genie API is called and result is cached

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade -r ../requirements.txt
# MAGIC %pip uninstall --quiet -y pyspark pyspark-connect
# MAGIC %restart_python

# COMMAND ----------

import sys, os, glob, subprocess

_wheels = glob.glob("../dist/dao_ai-*.whl") or glob.glob("../../artifacts/.internal/dao_ai-*.whl")
if _wheels:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall", _wheels[0]])
elif os.path.isdir("../src/dao_ai"):
    sys.path.insert(0, "../src")

try:
    from importlib.metadata import version
    print(f"dao-ai=={version('dao-ai')}")
except Exception:
    print("dao-ai==dev (source path)")

# COMMAND ----------

import nest_asyncio

nest_asyncio.apply()

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from loguru import logger

from dao_ai.logging import configure_logging

configure_logging(level="DEBUG")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Widgets
# MAGIC
# MAGIC Configure the Genie space, Lakebase project, and warehouse using the widgets below.

# COMMAND ----------

dbutils.widgets.text("space_id", "", "Genie Space ID")
dbutils.widgets.text("lakebase_project", "", "Lakebase Project")
dbutils.widgets.text("warehouse_id", "", "Warehouse ID")
dbutils.widgets.text("secret_scope", "retail_consumer_goods", "Secret Scope")
dbutils.widgets.text("client_id_secret", "RETAIL_AI_DATABRICKS_CLIENT_ID", "Client ID Secret Name")
dbutils.widgets.text("client_secret_secret", "RETAIL_AI_DATABRICKS_CLIENT_SECRET", "Client Secret Name")

# COMMAND ----------

# Get widget values
space_id: str = dbutils.widgets.get("space_id")
lakebase_project: str = dbutils.widgets.get("lakebase_project")
warehouse_id: str = dbutils.widgets.get("warehouse_id")
secret_scope: str = dbutils.widgets.get("secret_scope")
client_id_secret: str = dbutils.widgets.get("client_id_secret")
client_secret_secret: str = dbutils.widgets.get("client_secret_secret")

# Validate required parameters
if not space_id:
    raise ValueError("space_id widget is required")
if not lakebase_project:
    raise ValueError("lakebase_project widget is required")
if not warehouse_id:
    raise ValueError("warehouse_id widget is required")

print(f"Space ID: {space_id}")
print(f"Lakebase Project: {lakebase_project}")
print(f"Warehouse ID: {warehouse_id}")

# COMMAND ----------

# Get credentials from secrets
client_id: str = dbutils.secrets.get(secret_scope, client_id_secret)
client_secret: str = dbutils.secrets.get(secret_scope, client_secret_secret)

print("Credentials loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Cache Service
# MAGIC
# MAGIC Set up the chained cache architecture:
# MAGIC 1. Create the base Genie service
# MAGIC 2. Wrap with PostgresContextAwareGenieService for semantic caching
# MAGIC 3. Wrap with LRUCacheService for fast exact-match caching

# COMMAND ----------

from dao_ai.config import (
    DatabaseModel,
    GenieContextAwareCacheParametersModel,
    GenieLRUCacheParametersModel,
    WarehouseModel,
)
from dao_ai.genie import Genie, GenieResponse, GenieService, GenieServiceBase
from dao_ai.genie.cache import CacheResult, LRUCacheService
from dao_ai.genie.cache.context_aware import PostgresContextAwareGenieService

# COMMAND ----------

# Configure database connection (Lakebase)
database: DatabaseModel = DatabaseModel(
    project=lakebase_project,
    client_id=client_id,
    client_secret=client_secret,
)

# Configure warehouse for SQL execution
warehouse: WarehouseModel = WarehouseModel(warehouse_id=warehouse_id)

# Create base Genie service
genie: Genie = Genie(space_id=space_id)
genie_service: GenieServiceBase = GenieService(genie=genie)

print("Base Genie service created")

# COMMAND ----------

# Configure and wrap with context-aware cache
context_aware_cache_parameters: GenieContextAwareCacheParametersModel = (
    GenieContextAwareCacheParametersModel(
        database=database,
        warehouse=warehouse,
        time_to_live_seconds=86400 * 7,  # 7 days
        similarity_threshold=0.85,
        context_similarity_threshold=0.80,
        context_window_size=3,  # Number of previous messages to include in context
    )
)

genie_service = PostgresContextAwareGenieService(
    impl=genie_service,
    parameters=context_aware_cache_parameters,
).initialize()

print("Context-aware cache layer added")

# COMMAND ----------

# Configure and wrap with LRU cache
lru_cache_parameters: GenieLRUCacheParametersModel = GenieLRUCacheParametersModel(
    warehouse=warehouse,
    capacity=100,
    time_to_live_seconds=86400,  # 24 hours
)

genie_service = LRUCacheService(
    impl=genie_service,
    parameters=lru_cache_parameters,
)

print("LRU cache layer added")
print(f"\nFinal service chain:\n{genie_service}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect Service Chain
# MAGIC
# MAGIC The service is a chain of decorators. We can inspect each layer.

# COMMAND ----------

# View the full chain
print("Service chain:")
print(f"  Layer 1 (LRU): {type(genie_service).__name__}")
print(f"  Layer 2 (Context-Aware): {type(genie_service.impl).__name__}")
print(f"  Layer 3 (Genie): {type(genie_service.impl.impl).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## (Optional) Populate Cache from Existing Conversations
# MAGIC
# MAGIC If you have existing conversations in the Genie space, you can import them
# MAGIC into the cache to pre-populate it with embeddings.

# COMMAND ----------

# Uncomment to populate cache from existing Genie conversations
# This extracts questions and SQL from historical conversations

# from dao_ai.genie.cache.context_aware import PostgresContextAwareGenieService
#
# stats = PostgresContextAwareGenieService(
#     impl=genie_service,
#     parameters=context_aware_cache_parameters
# ).from_space(space_id=space_id, max_messages=50)
#
# print(f"Imported {stats} entries from Genie space")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ask Questions
# MAGIC
# MAGIC Now let's ask some questions and observe the caching behavior.
# MAGIC
# MAGIC The `CacheResult` includes:
# MAGIC - `response`: The GenieResponse with SQL and description
# MAGIC - `cache_hit`: Whether the result came from cache
# MAGIC - `served_by`: Which cache layer provided the result ("lru", "semantic", or None)
# MAGIC
# MAGIC **Note:** The cache only stores responses that include a SQL `query` attachment.
# MAGIC Pure narrative answers (e.g., "what tables are available?") are not cached because
# MAGIC there is no SQL to re-execute on a hit. Use aggregation/sampling prompts that force
# MAGIC Genie to generate SQL.

# COMMAND ----------

# First question - will be a cache miss (goes to Genie API)
# "Count the rows in each table" forces Genie to generate a SQL query
# (typically a UNION ALL of COUNT(*) or a query against information_schema)
result: CacheResult = genie_service.ask_question("Count the rows in each table.")

print(f"Cache hit: {result.cache_hit}")
print(f"Cache type: {result.served_by}")
print(f"SQL query: {result.response.query[:200] if result.response.query else '(none)'}")
print(f"\nResponse: {result.response}")

# COMMAND ----------

# Same question again - should be an LRU cache hit
result2: CacheResult = genie_service.ask_question("Count the rows in each table.")

print(f"Cache hit: {result2.cache_hit}")
print(f"Cache type: {result2.served_by}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-turn Conversation
# MAGIC
# MAGIC The context-aware cache tracks conversation history. Questions in the same
# MAGIC conversation share context, which affects cache matching.
# MAGIC
# MAGIC Follow-up prompts use relative references ("that table", "the largest one") so
# MAGIC Genie must use the conversation context to resolve them — this exercises the
# MAGIC context-aware cache's embeddings.

# COMMAND ----------

# Start a new conversation
conversation_id: str | None = None

# Question 1: Starting point - count rows per table (SQL required)
q1 = "Count the rows in each table."
result: CacheResult = genie_service.ask_question(q1)
conversation_id = result.response.conversation_id

print(f"Q1: {q1}")
print(f"  Cache hit: {result.cache_hit}, Type: {result.served_by}")
print(f"  Conversation ID: {conversation_id}")

# COMMAND ----------

# Question 2: Follow-up - identifies the largest table (SQL: ORDER BY + LIMIT)
q2 = "Which table has the most rows?"
result: CacheResult = genie_service.ask_question(q2, conversation_id=conversation_id)

print(f"Q2: {q2}")
print(f"  Cache hit: {result.cache_hit}, Type: {result.served_by}")

# COMMAND ----------

# Question 3: Another follow-up - samples rows from the resolved table (SQL: SELECT ... LIMIT)
q3 = "Show me the first 5 rows from that table."
result: CacheResult = genie_service.ask_question(q3, conversation_id=conversation_id)

print(f"Q3: {q3}")
print(f"  Cache hit: {result.cache_hit}, Type: {result.served_by}")

# COMMAND ----------

# Question 4: Continues the conversation - aggregation on the same table
q4 = "How many distinct rows does it have?"
result: CacheResult = genie_service.ask_question(q4, conversation_id=conversation_id)

print(f"Q4: {q4}")
print(f"  Cache hit: {result.cache_hit}, Type: {result.served_by}")

# COMMAND ----------

# Question 5: Context window may slide (if context_window_size=3) - total across all tables
q5 = "What is the total row count across all tables?"
result: CacheResult = genie_service.ask_question(q5, conversation_id=conversation_id)

print(f"Q5: {q5}")
print(f"  Cache hit: {result.cache_hit}, Type: {result.served_by}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cache Statistics
# MAGIC
# MAGIC You can inspect the cache to see what's stored.

# COMMAND ----------

# Get entries from the context-aware cache
# Access the PostgresContextAwareGenieService layer
context_aware_cache = genie_service.impl  # LRU -> Context-Aware

# Get cache entries (limit to 10 for display)
entries = context_aware_cache.get_entries(limit=10, include_embeddings=False)

print(f"Cache entries: {len(entries)}")
for i, entry in enumerate(entries[:5]):
    print(f"\n{i+1}. Question: {entry.get('question', 'N/A')[:60]}...")
    print(f"   Context: {entry.get('conversation_context', 'N/A')[:40]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup Widgets

# COMMAND ----------

# Uncomment to remove widgets
# dbutils.widgets.removeAll()
