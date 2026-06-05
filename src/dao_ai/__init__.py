"""
DAO AI - A framework for building AI agents with Databricks.

This module configures package-level settings including warning filters
for expected runtime warnings that don't indicate actual problems.
"""

import warnings


def _ensure_vector_search_client_index_alias() -> None:
    """Defensive shim for a transitive dep mismatch.

    ``databricks_openai.vector_search_retriever_tool`` does
    ``from databricks.vector_search.client import VectorSearchIndex`` at
    module load time. Newer ``databricks-vectorsearch`` releases provide
    this name as a convenience re-export from ``client.py``; older /
    Databricks-runtime-shipped builds do not.

    When dao-ai is loaded inside a Databricks serverless env whose
    pre-installed ``databricks-vectorsearch`` lacks the re-export, the
    import chain ``dao_ai.config → databricks_langchain → databricks_openai``
    breaks at module load. Re-pinning the wheel via ``requirements.txt``
    or the bundle env spec doesn't help when the env's package is
    pinned by the runtime.

    Patching the alias here, before any submodule import runs, makes
    the dao-ai import chain robust against either build. The shim is a
    no-op when the re-export is already present.
    """
    try:
        from databricks.vector_search import client as _client_mod
        from databricks.vector_search import index as _index_mod
    except Exception:
        # Vector search package not installed — nothing to patch.
        return
    if not hasattr(_client_mod, "VectorSearchIndex") and hasattr(
        _index_mod, "VectorSearchIndex"
    ):
        _client_mod.VectorSearchIndex = _index_mod.VectorSearchIndex


_ensure_vector_search_client_index_alias()

# Suppress Pydantic serialization warnings for Context objects during checkpointing.
# This warning occurs because LangGraph's checkpointer serializes the context_schema
# and Pydantic reports that serialization may not be as expected. This is benign
# since Context is only used at runtime and doesn't need to be persisted.
#
# The warning looks like:
# PydanticSerializationUnexpectedValue(Expected `none` - serialized value may not
# be as expected [field_name='context', input_value=Context(...), input_type=Context])
warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic serializer warnings.*",
    category=UserWarning,
)

# Also filter the specific PydanticSerializationUnexpectedValue warning
warnings.filterwarnings(
    "ignore",
    message=r".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)
