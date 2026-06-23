"""Unit tests for :mod:`dao_ai.apps.a2a.task_store`.

The Lakebase round-trip is exercised via integration tests; here we just
verify the :func:`build_task_store` selection logic against the new
:class:`A2ATaskStoreModel` config and that the
:class:`LakebaseTaskStore` constructor validates its inputs.
"""

import pytest
from a2a.server.tasks import InMemoryTaskStore

from dao_ai.apps.a2a.task_store import LakebaseTaskStore, build_task_store
from dao_ai.config import (
    A2AModel,
    A2ATaskStoreModel,
    AgentModel,
    AppConfig,
    AppModel,
    BackgroundModel,
    DatabaseModel,
    DeploymentTarget,
    InferenceEndpointModel,
)


def _agent() -> AgentModel:
    return AgentModel(
        name="greeter",
        description="says hi",
        model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
    )


def _database() -> DatabaseModel:
    return DatabaseModel(
        name="testdb",
        instance_name="test-instance",
        host="localhost",
        port=5432,
    )


def _config(
    *,
    a2a: A2AModel | None = None,
    background: BackgroundModel | None = None,
) -> AppConfig:
    extra: dict = {}
    if a2a is not None:
        extra["a2a"] = a2a
    return AppConfig(
        app=AppModel(
            name="dao-ai-test",
            description="test",
            deployment_target=DeploymentTarget.APPS,
            background=background,
            agents=[_agent()],
            **extra,
        ),
    )


@pytest.mark.unit
def test_build_task_store_defaults_to_in_memory():
    """No `a2a` block at all → in-memory."""
    cfg = _config(a2a=None)
    store = build_task_store(cfg)
    assert isinstance(store, InMemoryTaskStore)


@pytest.mark.unit
def test_build_task_store_explicit_empty_task_store_is_in_memory():
    """`a2a.task_store` present but with no database → in-memory."""
    cfg = _config(a2a=A2AModel(task_store=A2ATaskStoreModel()))
    store = build_task_store(cfg)
    assert isinstance(store, InMemoryTaskStore)


@pytest.mark.unit
def test_build_task_store_with_database_uses_lakebase_default_table():
    """`task_store.database` set → Lakebase with default table name."""
    cfg = _config(a2a=A2AModel(task_store=A2ATaskStoreModel(database=_database())))
    store = build_task_store(cfg)
    assert isinstance(store, LakebaseTaskStore)
    assert store.database.name == "testdb"
    assert store.table_name == "dao_ai_a2a_tasks"


@pytest.mark.unit
def test_build_task_store_honors_custom_table():
    """`task_store.table` overrides the default table name."""
    cfg = _config(
        a2a=A2AModel(
            task_store=A2ATaskStoreModel(
                database=_database(),
                table="my_custom_a2a_tasks",
            ),
        )
    )
    store = build_task_store(cfg)
    assert isinstance(store, LakebaseTaskStore)
    assert store.table_name == "my_custom_a2a_tasks"


@pytest.mark.unit
def test_build_task_store_is_independent_of_background():
    """A2A task store does NOT inherit a database from app.background.

    background has a database; task_store does not. Result: in-memory.
    """
    cfg = _config(
        a2a=A2AModel(),
        background=BackgroundModel(database=_database()),
    )
    store = build_task_store(cfg)
    assert isinstance(store, InMemoryTaskStore)


@pytest.mark.unit
def test_a2a_task_store_model_storage_type():
    """Sanity-check the `storage_type` property mirrors Checkpointer/Store."""
    from dao_ai.config import StorageType

    empty = A2ATaskStoreModel()
    assert empty.storage_type == StorageType.MEMORY

    persistent = A2ATaskStoreModel(database=_database())
    assert persistent.storage_type == StorageType.POSTGRES


@pytest.mark.unit
def test_lakebase_task_store_rejects_invalid_identifier():
    """SQL identifier validation guards against table-name injection."""
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
        LakebaseTaskStore(database=_database(), table_name="bad name; drop table users")
