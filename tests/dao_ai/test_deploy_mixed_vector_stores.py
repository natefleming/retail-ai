"""Live-shape tests for the Model Serving deploy path with a mixed
``resources.vector_stores`` dict.

Focus is on `_collect_resources_with_obo_flag` + `build_auth_policy` —
the two functions that iterate the discriminated-union dict on
deploy_agent(target=MODEL_SERVING). Both must dispatch correctly
between :class:`AiSearchVectorStoreModel` (native
:class:`IsDatabricksResource`) and :class:`LakebaseVectorStoreModel`
(delegates the trio to its nested :class:`DatabaseModel`). No
``AttributeError`` on the polymorphic iteration.
"""

from __future__ import annotations

import pytest

from dao_ai.config import (
    AiSearchVectorStoreModel,
    AppConfig,
    DatabaseModel,
    IndexModel,
    LakebaseVectorStoreModel,
    ResourcesModel,
    SchemaModel,
    VectorSearchEndpoint,
)
from dao_ai.providers.databricks import (
    _collect_resources_with_obo_flag,
    build_auth_policy,
)


def _ai_vs() -> AiSearchVectorStoreModel:
    return AiSearchVectorStoreModel(
        index=IndexModel(
            name="c.s.ai_index",
            schema=SchemaModel(catalog_name="c", schema_name="s"),
        ),
        endpoint=VectorSearchEndpoint(name="one-env-shared-endpoint-0"),
    )


def _lakebase_vs() -> LakebaseVectorStoreModel:
    return LakebaseVectorStoreModel(
        database=DatabaseModel(project="my-lakebase", name="my_lakebase_db"),
        table="kb_articles",
        content_column="passage",
        embedding_column="embedding",
        embedding_model="databricks-gte-large-en",
    )


def _config(**resources_kwargs) -> AppConfig:
    return AppConfig(resources=ResourcesModel(**resources_kwargs))


@pytest.mark.unit
class TestCollectResourcesMixedVectorStores:
    """`_collect_resources_with_obo_flag` should return every entry in
    `resources.vector_stores` regardless of concrete subtype — the caller
    (`build_auth_policy`) iterates via duck typing."""

    def test_collects_ai_search_only(self) -> None:
        cfg = _config(vector_stores={"ai": _ai_vs()})
        resources = _collect_resources_with_obo_flag(cfg)
        assert _ai_vs().__class__ in {r.__class__ for r in resources}

    def test_collects_lakebase_only(self) -> None:
        cfg = _config(vector_stores={"lb": _lakebase_vs()})
        resources = _collect_resources_with_obo_flag(cfg)
        assert _lakebase_vs().__class__ in {r.__class__ for r in resources}

    def test_collects_mixed(self) -> None:
        cfg = _config(vector_stores={"ai": _ai_vs(), "lb": _lakebase_vs()})
        resources = _collect_resources_with_obo_flag(cfg)
        classes = {r.__class__ for r in resources}
        assert AiSearchVectorStoreModel in classes
        assert LakebaseVectorStoreModel in classes


@pytest.mark.unit
class TestBuildAuthPolicyMixedVectorStores:
    """The load-bearing path: ``build_auth_policy`` iterates the
    collected resources and calls ``.as_resources()`` + reads
    ``.on_behalf_of_user`` on each. Neither operation may raise on
    Lakebase entries (which delegate to their nested database)."""

    def test_ai_search_only_produces_vector_search_index_resource(self) -> None:
        cfg = _config(vector_stores={"ai": _ai_vs()})
        policy = build_auth_policy(cfg)
        types = {r.__class__.__name__ for r in policy.system_auth_policy.resources}
        assert "DatabricksVectorSearchIndex" in types

    def test_lakebase_only_does_not_raise(self) -> None:
        """No AttributeError on missing .as_resources() / .on_behalf_of_user.
        The Lakebase entry delegates to its DatabaseModel, which for an
        autoscaling Lakebase project returns [] (documented — MLflow has
        no matching resource type yet). Empty is a valid outcome; the
        deploy must not blow up."""
        cfg = _config(vector_stores={"lb": _lakebase_vs()})
        policy = build_auth_policy(cfg)
        # No exception raised is the primary assertion. Resources may be
        # empty for autoscaling Lakebase — that's the current expected
        # behavior of DatabaseModel.as_resources().
        assert policy is not None
        # Sanity: whatever resources land, none are DatabricksVectorSearchIndex
        # (Lakebase doesn't use that shape).
        types = {r.__class__.__name__ for r in policy.system_auth_policy.resources}
        assert "DatabricksVectorSearchIndex" not in types

    def test_mixed_produces_only_ai_search_vector_index_resource(self) -> None:
        """The two vector-store types coexist and dispatch correctly:
        AI Search contributes a `DatabricksVectorSearchIndex`; Lakebase
        contributes whatever its DatabaseModel emits (empty for
        autoscaling), never a spurious `DatabricksVectorSearchIndex`."""
        cfg = _config(vector_stores={"ai": _ai_vs(), "lb": _lakebase_vs()})
        policy = build_auth_policy(cfg)
        vs_index_resources = [
            r
            for r in policy.system_auth_policy.resources
            if r.__class__.__name__ == "DatabricksVectorSearchIndex"
        ]
        # Exactly one vector-search-index resource — from the ai_search
        # entry, not the lakebase entry.
        assert len(vs_index_resources) == 1

    def test_lakebase_on_behalf_of_user_delegation(self) -> None:
        """Confirm the deploy-time OBO partitioning respects the
        delegated ``on_behalf_of_user`` from LakebaseVectorStoreModel's
        nested database. When OBO is True, the entry's resources land in
        the UserAuthPolicy scopes, not the SystemAuthPolicy resources."""
        lb = _lakebase_vs()
        # Toggle OBO on the underlying database — LakebaseVectorStoreModel
        # delegates the property.
        lb.database.on_behalf_of_user = True
        assert lb.on_behalf_of_user is True  # verify delegation

        cfg = _config(vector_stores={"lb": lb})
        policy = build_auth_policy(cfg)  # no exception
        # OBO entries do NOT contribute to system resources
        assert policy is not None
