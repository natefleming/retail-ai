"""Tests for AuditModel configuration."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from dao_ai.config import AuditModel, DatabaseModel


class TestAuditModelDefaults:
    """Tests for AuditModel default field values and construction."""

    def test_minimum_config_uses_defaults(self) -> None:
        """AuditModel requires only a database; table and TTL take defaults."""
        model = AuditModel(database=DatabaseModel(project="test-lakebase"))
        assert model.table == "audit_receipts"
        assert model.nonce_ttl_seconds == 300

    def test_custom_table_name(self) -> None:
        model = AuditModel(
            database=DatabaseModel(project="test-lakebase"),
            table="custom_receipts",
        )
        assert model.table == "custom_receipts"

    def test_custom_nonce_ttl(self) -> None:
        model = AuditModel(
            database=DatabaseModel(project="test-lakebase"),
            nonce_ttl_seconds=600,
        )
        assert model.nonce_ttl_seconds == 600


class TestAuditModelValidation:
    """Tests for AuditModel validation constraints."""

    def test_database_is_required(self) -> None:
        with pytest.raises(ValidationError):
            AuditModel()  # type: ignore[call-arg]

    def test_extra_fields_forbidden(self) -> None:
        """extra='forbid' — unknown fields must reject."""
        with pytest.raises(ValidationError):
            AuditModel(
                database=DatabaseModel(project="test-lakebase"),
                unknown_field="value",  # type: ignore[call-arg]
            )

    def test_nonce_ttl_below_minimum(self) -> None:
        """Nonce TTL below 30s is rejected (too short for realistic UI flows)."""
        with pytest.raises(ValidationError):
            AuditModel(
                database=DatabaseModel(project="test-lakebase"),
                nonce_ttl_seconds=10,
            )

    def test_nonce_ttl_above_maximum(self) -> None:
        """Nonce TTL above 3600s is rejected (approvals should not linger for hours)."""
        with pytest.raises(ValidationError):
            AuditModel(
                database=DatabaseModel(project="test-lakebase"),
                nonce_ttl_seconds=7200,
            )

    def test_nonce_ttl_at_boundaries(self) -> None:
        """Boundary values 30 and 3600 are accepted."""
        low = AuditModel(
            database=DatabaseModel(project="test-lakebase"),
            nonce_ttl_seconds=30,
        )
        assert low.nonce_ttl_seconds == 30

        high = AuditModel(
            database=DatabaseModel(project="test-lakebase"),
            nonce_ttl_seconds=3600,
        )
        assert high.nonce_ttl_seconds == 3600


class TestAuditModelSerialization:
    """Tests for round-trip serialization of AuditModel via YAML."""

    def test_yaml_round_trip(self) -> None:
        """model_dump → yaml → model_validate produces an equal model."""
        original = AuditModel(
            database=DatabaseModel(project="round-trip-lakebase"),
            table="audit_rows",
            nonce_ttl_seconds=180,
        )

        dumped = original.model_dump(mode="json")
        rehydrated_dict = yaml.safe_load(yaml.safe_dump(dumped))
        rehydrated = AuditModel.model_validate(rehydrated_dict)

        assert rehydrated.table == original.table
        assert rehydrated.nonce_ttl_seconds == original.nonce_ttl_seconds
        assert rehydrated.database.project == original.database.project


class TestBaseFunctionModelAuditField:
    """Tests for the optional audit field on BaseFunctionModel subclasses."""

    def test_python_function_without_audit_defaults_none(self) -> None:
        """The audit field is absent by default — feature is fully opt-in."""
        from dao_ai.config import PythonFunctionModel

        fn = PythonFunctionModel(name="pkg.module.tool")
        assert fn.audit is None
        assert fn.human_in_the_loop is None

    def test_python_function_with_audit(self) -> None:
        """AuditModel can be attached to a function alongside HITL."""
        from dao_ai.config import HumanInTheLoopModel, PythonFunctionModel

        audit = AuditModel(database=DatabaseModel(project="fn-audit-lakebase"))
        fn = PythonFunctionModel(
            name="pkg.module.tool",
            audit=audit,
            human_in_the_loop=HumanInTheLoopModel(review_prompt="Approve?"),
        )
        assert fn.audit is audit
        assert fn.human_in_the_loop is not None

    def test_python_function_audit_without_hitl(self) -> None:
        """audit is independent of human_in_the_loop — either can be set alone."""
        from dao_ai.config import PythonFunctionModel

        audit = AuditModel(database=DatabaseModel(project="fn-audit-lakebase"))
        fn = PythonFunctionModel(name="pkg.module.tool", audit=audit)
        assert fn.audit is audit
        assert fn.human_in_the_loop is None


class TestAuditModelYamlAnchorScenario:
    """
    Exercises the recommended YAML anchor pattern from the plan: a single
    audit block referenced from multiple audited tools.
    """

    _YAML = """
        function_a:
          type: python
          name: pkg.module.tool_a
          audit: &audit
            database:
              project: shared-lakebase
            table: shared_receipts
        function_b:
          type: python
          name: pkg.module.tool_b
          audit: *audit
        function_c:
          type: python
          name: pkg.module.tool_c
          # no audit block — this tool is not audited
    """

    def test_anchor_shared_across_tools(self) -> None:
        from dao_ai.config import PythonFunctionModel

        parsed = yaml.safe_load(self._YAML)
        fn_a = PythonFunctionModel.model_validate(parsed["function_a"])
        fn_b = PythonFunctionModel.model_validate(parsed["function_b"])
        fn_c = PythonFunctionModel.model_validate(parsed["function_c"])

        assert fn_a.audit is not None
        assert fn_b.audit is not None
        assert fn_c.audit is None

        # Both audited tools resolve to the same table + database identity.
        assert fn_a.audit.table == fn_b.audit.table == "shared_receipts"
        assert (
            fn_a.audit.database.project == fn_b.audit.database.project == "shared-lakebase"
        )
