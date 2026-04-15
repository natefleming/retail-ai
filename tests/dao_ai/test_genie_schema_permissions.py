"""
Tests for Genie table permission handling.

Covers:
- TableModel.api_scopes returning sql.statement-execution for OBO
- OBO tables contributing the sql user API scope
"""

from unittest.mock import MagicMock

import pytest

from dao_ai.config import TableModel


@pytest.mark.unit
class TestTableModelApiScopes:
    """TableModel.api_scopes should return sql.statement-execution for OBO."""

    def test_api_scopes_returns_sql_statement_execution(self):
        table = TableModel(name="catalog.schema.my_table")
        assert table.api_scopes == ["sql.statement-execution"]

    def test_api_scopes_is_sequence(self):
        table = TableModel(name="catalog.schema.t")
        assert hasattr(table.api_scopes, "__len__")
        assert len(table.api_scopes) == 1


@pytest.mark.unit
class TestUserApiScopesWithTables:
    """Verify that OBO tables contribute the sql user API scope."""

    def test_obo_tables_add_sql_scope(self):
        from dao_ai.apps.resources import generate_user_api_scopes

        config = MagicMock()
        config.resources = MagicMock()

        # Set up a table with on_behalf_of_user=True
        table = TableModel(name="cat.sch.t1", on_behalf_of_user=True)

        config.resources.llms = {}
        config.resources.vector_stores = {}
        config.resources.warehouses = {}
        config.resources.genie_rooms = {}
        config.resources.volumes = {}
        config.resources.functions = {}
        config.resources.connections = {}
        config.resources.databases = {}
        config.resources.tables = {"t1": table}

        scopes = generate_user_api_scopes(config)
        assert "sql" in scopes
        assert "catalog.catalogs:read" in scopes
        assert "catalog.schemas:read" in scopes
        assert "catalog.tables:read" in scopes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
