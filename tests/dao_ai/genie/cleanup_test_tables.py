#!/usr/bin/env python
"""
Utility script to clean up test tables from Lakebase.

This script removes test tables that were created by integration tests
but not properly cleaned up. Run this periodically to avoid accumulating
test artifacts in the database.

Usage:
    python -m tests.dao_ai.genie.cleanup_test_tables

Or import and use programmatically:
    from tests.dao_ai.genie.cleanup_test_tables import cleanup_test_tables
    cleanup_test_tables()
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Test table name patterns to clean up
TEST_TABLE_PATTERNS = [
    "test_from_space_cache_%",
    "test_from_space_prompts_%",
    "test_context_aware_cache_%",
    "test_prompt_history_%",
    "test_cache_%",
]


def cleanup_test_tables(
    instance_name: str | None = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Clean up test tables from Lakebase.

    Args:
        instance_name: Lakebase instance name. If None, uses LAKEBASE_INSTANCE_NAME
                       env var or defaults to "retail-consumer-goods".
        dry_run: If True, only list tables without dropping them.
        verbose: If True, print progress to stdout.

    Returns:
        Dict with 'tables_found', 'tables_dropped', and 'errors' keys.
    """
    # Import here to avoid issues when module is imported in non-Databricks env
    from databricks.sdk import WorkspaceClient

    from dao_ai.config import DatabaseModel

    # Get instance name from args, env, or default
    if instance_name is None:
        instance_name = os.environ.get(
            "LAKEBASE_INSTANCE_NAME", "retail-consumer-goods"
        )

    if verbose:
        print(f"Connecting to Lakebase instance: {instance_name}")

    # Create database connection
    ws = WorkspaceClient()
    database = DatabaseModel(
        instance_name=instance_name,
        workspace_client=ws,
    )

    results: dict[str, Any] = {
        "tables_found": [],
        "tables_dropped": [],
        "errors": [],
        "instance_name": instance_name,
        "dry_run": dry_run,
    }

    try:
        pool = database.get_pool()

        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Build query to find all test tables
                pattern_conditions = " OR ".join(
                    f"tablename LIKE '{pattern}'" for pattern in TEST_TABLE_PATTERNS
                )
                find_tables_sql = f"""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public' 
                      AND ({pattern_conditions})
                    ORDER BY tablename
                """

                cur.execute(find_tables_sql)
                rows = cur.fetchall()
                tables = [row["tablename"] for row in rows]
                results["tables_found"] = tables

                if verbose:
                    print(f"Found {len(tables)} test tables")

                if not tables:
                    if verbose:
                        print("No test tables to clean up.")
                    return results

                if dry_run:
                    if verbose:
                        print("\n[DRY RUN] Would drop these tables:")
                        for table in tables:
                            print(f"  - {table}")
                    return results

                # Drop each table
                for table in tables:
                    try:
                        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                        results["tables_dropped"].append(table)
                        if verbose:
                            print(f"  Dropped: {table}")
                    except Exception as e:
                        error_msg = f"Failed to drop {table}: {e}"
                        results["errors"].append(error_msg)
                        if verbose:
                            print(f"  ERROR: {error_msg}")

                if verbose:
                    print(
                        f"\nCleanup complete: {len(results['tables_dropped'])} tables dropped, "
                        f"{len(results['errors'])} errors"
                    )

    except Exception as e:
        error_msg = f"Database connection error: {e}"
        results["errors"].append(error_msg)
        if verbose:
            print(f"ERROR: {error_msg}")

    return results


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean up test tables from Lakebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--instance",
        "-i",
        help="Lakebase instance name (default: from LAKEBASE_INSTANCE_NAME env or 'retail-consumer-goods')",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="List tables without dropping them",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    results = cleanup_test_tables(
        instance_name=args.instance,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    # Return non-zero exit code if there were errors
    return 1 if results["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
