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
import glob, os

from packaging.version import Version

# Newest by *version*, not by filename: a lexical sort puts 0.2.8 above
# 0.2.10. ``Version`` also parses a dev wheel's ``+local`` tag correctly.
def _wheel_version(wheel: str) -> Version:
    return Version(os.path.basename(wheel).split("-")[1])

_wheels = sorted(glob.glob("../dist/dao_ai-*.whl"), key=_wheel_version, reverse=True)
_dao_ai_dep = _wheels[0] if _wheels else "dao-ai"

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

dbutils.widgets.text(name="config-path", defaultValue="")

# There is no `../config` discovery fallback. That directory exists only in the
# staged bundle layout, and the bundle stages exactly one config — the same one
# the job passes here — so discovery could only ever guess, and guessing is how
# the wrong config gets loaded. `config-path` is the single input.
widget_path: str | None = dbutils.widgets.get("config-path") or None
if not widget_path:
    raise ValueError(
        "No config to load: the `config-path` widget is empty. In a staged "
        "pipeline bundle the config sits beside this notebook under `../config/` "
        "and the job always passes it; running this notebook by hand, set "
        "`config-path` to an absolute workspace path, for example "
        "`/Workspace/Users/you@example.com/dao-ai/examples/04_genie/genie_basic.yaml`."
    )

config_path: str = widget_path

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
