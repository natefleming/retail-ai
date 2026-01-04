"""
Unit tests for MCP tool filtering functionality.

Tests the pattern matching and filtering logic for including/excluding
tools from MCP servers.
"""

from dao_ai.tools.mcp import _matches_pattern, _should_include_tool


class TestMatchesPattern:
    """Tests for glob pattern matching functionality."""

    def test_exact_match(self):
        """Test exact string matching."""
        assert _matches_pattern("execute_query", ["execute_query"])
        assert _matches_pattern("execute_query", ["other", "execute_query"])
        assert not _matches_pattern("execute_query", ["list_tables"])
        assert not _matches_pattern("execute_query", ["execute"])

    def test_glob_star_matches_any_characters(self):
        """Test * pattern matches zero or more characters."""
        assert _matches_pattern("query_sales", ["query_*"])
        assert _matches_pattern("query_", ["query_*"])
        assert _matches_pattern("query_sales_data_by_region", ["query_*"])
        assert not _matches_pattern("execute_query", ["query_*"])
        assert not _matches_pattern("sales_query", ["query_*"])

    def test_glob_star_multiple_patterns(self):
        """Test * pattern in multiple positions."""
        assert _matches_pattern("get_sales_data", ["get_*_data"])
        assert _matches_pattern("get__data", ["get_*_data"])
        assert not _matches_pattern(
            "get_data", ["get_*_data"]
        )  # Need at least one char

    def test_glob_question_matches_single_character(self):
        """Test ? pattern matches exactly one character."""
        assert _matches_pattern("tool_a", ["tool_?"])
        assert _matches_pattern("tool_1", ["tool_?"])
        assert not _matches_pattern("tool_ab", ["tool_?"])
        assert not _matches_pattern("tool_", ["tool_?"])
        assert not _matches_pattern("tool", ["tool_?"])

    def test_glob_bracket_matches_character_set(self):
        """Test [abc] pattern matches any char in set."""
        assert _matches_pattern("tool_a", ["tool_[abc]"])
        assert _matches_pattern("tool_b", ["tool_[abc]"])
        assert _matches_pattern("tool_c", ["tool_[abc]"])
        assert not _matches_pattern("tool_d", ["tool_[abc]"])
        assert not _matches_pattern("tool_ab", ["tool_[abc]"])

    def test_glob_bracket_negation_matches_not_in_set(self):
        """Test [!abc] pattern matches any char NOT in set."""
        assert _matches_pattern("tool_d", ["tool_[!abc]"])
        assert _matches_pattern("tool_1", ["tool_[!abc]"])
        assert not _matches_pattern("tool_a", ["tool_[!abc]"])
        assert not _matches_pattern("tool_b", ["tool_[!abc]"])

    def test_complex_patterns(self):
        """Test complex glob patterns."""
        # Match all query tools except specific ones
        assert _matches_pattern("query_sales", ["query_*"])
        assert _matches_pattern("query_inventory", ["query_*"])

        # Match all operations with specific prefixes
        assert _matches_pattern("get_user_data", ["get_*", "list_*", "fetch_*"])
        assert _matches_pattern("list_tables", ["get_*", "list_*", "fetch_*"])
        assert not _matches_pattern("delete_user", ["get_*", "list_*", "fetch_*"])

    def test_empty_patterns(self):
        """Test behavior with empty pattern list."""
        assert not _matches_pattern("any_tool", [])

    def test_case_sensitive(self):
        """Test that pattern matching is case-sensitive."""
        assert _matches_pattern("query_sales", ["query_*"])
        assert not _matches_pattern("Query_sales", ["query_*"])
        assert not _matches_pattern("QUERY_SALES", ["query_*"])


class TestShouldIncludeTool:
    """Tests for tool inclusion/exclusion decision logic."""

    def test_no_filters_includes_all(self):
        """Test default behavior: include all tools when no filters specified."""
        assert _should_include_tool("any_tool", None, None)
        assert _should_include_tool("query_sales", None, None)
        assert _should_include_tool("drop_table", None, None)

    def test_include_list_allowlist(self):
        """Test include_tools as an allowlist."""
        assert _should_include_tool("query_sales", ["query_*"], None)
        assert _should_include_tool("query_inventory", ["query_*"], None)
        assert not _should_include_tool("list_tables", ["query_*"], None)
        assert not _should_include_tool("drop_table", ["query_*"], None)

    def test_include_list_exact_matches(self):
        """Test include_tools with exact tool names."""
        include = ["execute_query", "list_tables", "describe_table"]

        assert _should_include_tool("execute_query", include, None)
        assert _should_include_tool("list_tables", include, None)
        assert _should_include_tool("describe_table", include, None)
        assert not _should_include_tool("drop_table", include, None)

    def test_include_list_mixed_exact_and_patterns(self):
        """Test include_tools with mix of exact names and patterns."""
        include = ["execute_query", "list_*", "get_?_data"]

        assert _should_include_tool("execute_query", include, None)
        assert _should_include_tool("list_tables", include, None)
        assert _should_include_tool("list_schemas", include, None)
        assert _should_include_tool("get_a_data", include, None)
        assert not _should_include_tool("drop_table", include, None)
        assert not _should_include_tool("get_abc_data", include, None)

    def test_exclude_list_denylist(self):
        """Test exclude_tools as a denylist."""
        assert not _should_include_tool("drop_table", None, ["drop_*"])
        assert not _should_include_tool("drop_schema", None, ["drop_*"])
        assert _should_include_tool("query_sales", None, ["drop_*"])
        assert _should_include_tool("list_tables", None, ["drop_*"])

    def test_exclude_list_exact_matches(self):
        """Test exclude_tools with exact tool names."""
        exclude = ["drop_table", "truncate_table", "execute_ddl"]

        assert not _should_include_tool("drop_table", None, exclude)
        assert not _should_include_tool("truncate_table", None, exclude)
        assert not _should_include_tool("execute_ddl", None, exclude)
        assert _should_include_tool("execute_query", None, exclude)

    def test_exclude_list_patterns(self):
        """Test exclude_tools with patterns."""
        exclude = ["drop_*", "delete_*", "truncate_*"]

        assert not _should_include_tool("drop_table", None, exclude)
        assert not _should_include_tool("delete_record", None, exclude)
        assert not _should_include_tool("truncate_table", None, exclude)
        assert _should_include_tool("query_sales", None, exclude)

    def test_exclude_overrides_include(self):
        """Test that exclude_tools takes precedence over include_tools."""
        include = ["query_*"]
        exclude = ["*_sensitive"]

        # Matches include but also matches exclude - should be excluded
        assert not _should_include_tool("query_sensitive", include, exclude)

        # Matches include but not exclude - should be included
        assert _should_include_tool("query_sales", include, exclude)

        # Doesn't match include - should be excluded
        assert not _should_include_tool("list_tables", include, exclude)

    def test_exclude_overrides_include_complex(self):
        """Test complex interaction of include and exclude."""
        include = ["query_*", "get_*", "list_*"]
        exclude = ["*_admin", "*_secret"]

        # Should be included: matches include, doesn't match exclude
        assert _should_include_tool("query_sales", include, exclude)
        assert _should_include_tool("get_inventory", include, exclude)
        assert _should_include_tool("list_tables", include, exclude)

        # Should be excluded: matches both include and exclude
        assert not _should_include_tool("query_admin", include, exclude)
        assert not _should_include_tool("get_secret", include, exclude)
        assert not _should_include_tool("list_admin", include, exclude)

        # Should be excluded: doesn't match include
        assert not _should_include_tool("drop_table", include, exclude)

    def test_empty_lists_treated_as_no_filter(self):
        """Test that empty lists are treated as no filter (falsy values)."""
        # Empty include list is falsy, treated as None (no filter)
        assert _should_include_tool("any_tool", [], None)

        # Empty exclude list is falsy, treated as None (no filter)
        assert _should_include_tool("any_tool", None, [])

        # Both empty: treated as no filters at all
        assert _should_include_tool("any_tool", [], [])

    def test_multiple_patterns_in_include(self):
        """Test multiple patterns in include list."""
        include = ["query_*", "get_*", "list_*"]

        assert _should_include_tool("query_sales", include, None)
        assert _should_include_tool("get_inventory", include, None)
        assert _should_include_tool("list_tables", include, None)
        assert not _should_include_tool("drop_table", include, None)
        assert not _should_include_tool("execute_query", include, None)

    def test_multiple_patterns_in_exclude(self):
        """Test multiple patterns in exclude list."""
        exclude = ["drop_*", "delete_*", "truncate_*", "execute_ddl"]

        assert not _should_include_tool("drop_table", None, exclude)
        assert not _should_include_tool("delete_record", None, exclude)
        assert not _should_include_tool("truncate_table", None, exclude)
        assert not _should_include_tool("execute_ddl", None, exclude)
        assert _should_include_tool("execute_query", None, exclude)
        assert _should_include_tool("query_sales", None, exclude)

    def test_real_world_sql_safe_tools(self):
        """Test real-world scenario: SQL tools with read-only filtering."""
        include = ["execute_query", "list_*", "describe_*", "get_*"]
        exclude = ["execute_ddl"]

        # Should include read operations
        assert _should_include_tool("execute_query", include, exclude)
        assert _should_include_tool("list_tables", include, exclude)
        assert _should_include_tool("describe_table", include, exclude)
        assert _should_include_tool("get_schema", include, exclude)

        # Should exclude write operations
        assert not _should_include_tool("drop_table", include, exclude)
        assert not _should_include_tool("create_table", include, exclude)
        assert not _should_include_tool("execute_ddl", include, exclude)

    def test_real_world_query_tools_only(self):
        """Test real-world scenario: Only query/read tools allowed."""
        include = ["query_*", "list_*", "get_*", "describe_*"]
        exclude = ["*_admin", "*_sensitive"]

        # Should include read operations
        assert _should_include_tool("query_sales", include, exclude)
        assert _should_include_tool("list_inventory", include, exclude)
        assert _should_include_tool("get_product", include, exclude)
        assert _should_include_tool("describe_schema", include, exclude)

        # Should exclude admin/sensitive
        assert not _should_include_tool("query_admin", include, exclude)
        assert not _should_include_tool("get_sensitive", include, exclude)

        # Should exclude write operations
        assert not _should_include_tool("delete_record", include, exclude)
        assert not _should_include_tool("update_table", include, exclude)

    def test_real_world_block_dangerous_operations(self):
        """Test real-world scenario: Block dangerous operations."""
        exclude = ["drop_*", "delete_*", "truncate_*", "alter_*", "execute_ddl"]

        # Should allow safe operations
        assert _should_include_tool("query_sales", None, exclude)
        assert _should_include_tool("list_tables", None, exclude)
        assert _should_include_tool("execute_query", None, exclude)

        # Should block dangerous operations
        assert not _should_include_tool("drop_table", None, exclude)
        assert not _should_include_tool("delete_data", None, exclude)
        assert not _should_include_tool("truncate_table", None, exclude)
        assert not _should_include_tool("alter_schema", None, exclude)
        assert not _should_include_tool("execute_ddl", None, exclude)
