"""Unit tests for :mod:`dao_ai.apps.a2a.task_store`.

The Lakebase round-trip is exercised via integration tests; here we just
verify the :func:`build_task_store` selection logic and that the
:class:`LakebaseTaskStore` constructor validates its inputs.
"""

import pytest
from a2a.server.tasks import InMemoryTaskStore

from dao_ai.apps.a2a.task_store import LakebaseTaskStore, build_task_store
from dao_ai.config import (
    A2AModel,
    AgentModel,
    AppConfig,
    AppModel,
    DatabaseModel,
    DeploymentTarget,
    InferenceEndpointModel,
    LongRunningModel,
)


def _agent() -> AgentModel:
    return AgentModel(
        name="greeter",
        description="says hi",
        model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
    )


def _config(*, a2a: A2AModel | None, long_running: LongRunningModel | None) -> AppConfig:
    return AppConfig(
        app=AppModel(
            name="dao-ai-test",
            description="test",
            deployment_target=DeploymentTarget.APPS,
            a2a=a2a,
            long_running=long_running,
            agents=[_agent()],
        ),
    )


def _long_running() -> LongRunningModel:
    return LongRunningModel(
        database=DatabaseModel(
            name="testdb",
            instance_name="test-instance",
            host="localhost",
            port=5432,
        )
    )


@pytest.mark.unit
def test_build_task_store_auto_no_long_running_uses_memory():
    cfg = _config(a2a=None, long_running=None)
    store = build_task_store(cfg)
    assert isinstance(store, InMemoryTaskStore)


@pytest.mark.unit
def test_build_task_store_auto_with_long_running_uses_lakebase():
    cfg = _config(a2a=None, long_running=_long_running())
    store = build_task_store(cfg)
    assert isinstance(store, LakebaseTaskStore)
    assert store.table_name == "dao_ai_a2a_tasks"


@pytest.mark.unit
def test_build_task_store_in_memory_forces_memory_even_with_lakebase():
    cfg = _config(a2a=A2AModel(task_store="in_memory"), long_running=_long_running())
    store = build_task_store(cfg)
    assert isinstance(store, InMemoryTaskStore)


@pytest.mark.unit
def test_build_task_store_lakebase_without_long_running_raises():
    cfg = _config(a2a=A2AModel(task_store="lakebase"), long_running=None)
    with pytest.raises(ValueError, match="app.long_running"):
        build_task_store(cfg)


@pytest.mark.unit
def test_build_task_store_honors_custom_table_name():
    cfg = _config(
        a2a=A2AModel(task_store="lakebase", tasks_table_name="my_custom_a2a_tasks"),
        long_running=_long_running(),
    )
    store = build_task_store(cfg)
    assert isinstance(store, LakebaseTaskStore)
    assert store.table_name == "my_custom_a2a_tasks"


@pytest.mark.unit
def test_lakebase_task_store_rejects_invalid_identifier():
    """SQL identifier validation guards against table-name injection."""
    db = DatabaseModel(
        name="testdb",
        instance_name="test-instance",
        host="localhost",
        port=5432,
    )
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
        LakebaseTaskStore(database=db, table_name="bad name; drop table users")
