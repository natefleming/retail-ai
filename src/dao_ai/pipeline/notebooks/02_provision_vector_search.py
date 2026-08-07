# Databricks notebook source
# Dependency bootstrap. Install dao-ai (which pulls its own transitive deps) via
# uv — the newest bundled ../dist wheel in development mode, else the published
# PyPI package. In a deployed job the serverless environment has already
# installed it; this reinstall is harmless. ``%restart_python`` makes the freshly
# installed package importable in the cells below.
# No extras suffix: this provisioning notebook only calls core APIs. Notebooks
# that build the agent graph (07_deploy_agent, 09_run_evaluation) install
# ``[all]``; 01_ingest_and_transform installs ``[excel]``. The install spec is
# single-quoted in the magic so a dev wheel's ``+local`` version tag and any
# ``[extras]`` survive shell glob/bracket expansion.
import glob

_dao_ai_dep = next(
    iter(sorted(glob.glob("../dist/dao_ai-*.whl"), reverse=True)), "dao-ai"
)

# MAGIC %uv pip install --quiet '{_dao_ai_dep}'
# MAGIC %restart_python

# COMMAND ----------

# Record the installed dao-ai version plus the key libraries resolved under
# it, so each run's logs capture exactly what was installed. Alphabetical; the
# list is short and hand-curated.
from importlib.metadata import version

print(f"dao-ai=={version('dao-ai')}")
print(f"databricks-langchain=={version('databricks-langchain')}")
print(f"databricks-sdk=={version('databricks-sdk')}")
print(f"langchain=={version('langchain')}")
print(f"langgraph=={version('langgraph')}")
print(f"mlflow=={version('mlflow')}")

# COMMAND ----------

from dao_ai.utils import find_config_files

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")

config_files: list[str] = find_config_files("../config")
dbutils.widgets.dropdown(
    name="config-paths", choices=config_files, defaultValue=next(iter(config_files), "")
)

explicit_path: str | None = dbutils.widgets.get("config-path") or None
discovered_path: str | None = dbutils.widgets.get("config-paths") or None

# An explicit `config-path` always wins; the `../config` dropdown is the fallback
# for an interactive run. Re-binding `config_path` from `str | None` to `str` would
# make the annotation a lie (and hide that `from_file` rejects None), so the inputs
# get their own names and the result is annotated once.
resolved_path: str | None = explicit_path or discovered_path
if not resolved_path:
    raise ValueError(
        "No config to load: the `config-path` widget is empty and no YAML was "
        "found under ../config. Set `config-path` to the staged config (the "
        "pipeline bundle always passes it)."
    )

config_path: str = resolved_path

print(config_path)

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from dao_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

from dao_ai.config import VectorStoreModel

vector_stores: dict[str, VectorStoreModel] = config.resources.vector_stores

for _, vector_store in vector_stores.items():
    vector_store: VectorStoreModel

    print(f"vector_store: {vector_store}")
    vector_store.create()


# COMMAND ----------

from typing import Any, Dict

from databricks.ai_search.index import VectorSearchIndex

from dao_ai.config import AiSearchRetrieverModel

question: str = "How many grills do we have in stock?"

for name, retriever in config.retrievers.items():
    # AppConfig.retrievers is a discriminated union of AiSearchRetrieverModel
    # and LakebaseRetrieverModel. This notebook is Vector Search only, so
    # skip Lakebase entries.
    if not isinstance(retriever, AiSearchRetrieverModel):
        continue
    index: VectorSearchIndex = retriever.vector_store.as_index()
    k: int = 3

    search_results: Dict[str, Any] = index.similarity_search(
        query_text=question,
        columns=retriever.columns,
        **retriever.search_parameters.model_dump(),
    )

    chunks: list[str] = search_results.get("result", {}).get("data_array", [])
    print(len(chunks))
    print(chunks)

# COMMAND ----------

from typing import Sequence

from databricks_langchain import DatabricksVectorSearch
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStore

content = "What grills do you have in stock?"
for name, retriever in config.retrievers.items():
    vector_search: VectorStore = DatabricksVectorSearch(
        endpoint=retriever.vector_store.endpoint.name,
        index_name=retriever.vector_store.index.full_name,
        columns=retriever.columns,
        client_args={},
    )

    documents: Sequence[Document] = vector_search.similarity_search(
        query=content, **retriever.search_parameters.model_dump()
    )
    print(len(documents))
