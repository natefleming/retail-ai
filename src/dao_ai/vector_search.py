from databricks.vector_search.client import VectorSearchClient


def endpoint_exists(vsc: VectorSearchClient, vs_endpoint_name: str) -> bool:
    """
    Check if a Vector Search endpoint exists in the Databricks workspace.

    This utility function verifies whether a given Vector Search endpoint is already
    provisioned, handling rate limit errors gracefully to avoid workflow disruptions.

    Args:
        vsc: Databricks Vector Search client instance
        vs_endpoint_name: Name of the Vector Search endpoint to check

    Returns:
        True if the endpoint exists, False otherwise

    Raises:
        Exception: If an unexpected error occurs while checking endpoint existence
                  (except for rate limit errors, which are handled gracefully)
    """
    try:
        # Retrieve all endpoints and check if the target endpoint name is in the list
        return vs_endpoint_name in [
            e["name"] for e in vsc.list_endpoints().get("endpoints", [])
        ]
    except Exception as e:
        # Special handling for rate limit errors to prevent workflow failures
        if "REQUEST_LIMIT_EXCEEDED" in str(e):
            print(
                "WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error."
            )
            # Assume endpoint exists to avoid disrupting the workflow
            return True
        else:
            # Re-raise other unexpected errors
            raise e


def index_exists(
    vsc: VectorSearchClient, endpoint_name: str, index_full_name: str
) -> bool:
    """
    Check if a Vector Search index exists on a specific endpoint.

    This utility function verifies whether a given Vector Search index is already
    created on the specified endpoint, handling non-existence errors gracefully.

    Args:
        vsc: Databricks Vector Search client instance
        endpoint_name: Name of the Vector Search endpoint to check
        index_full_name: Fully qualified name of the index (catalog.schema.table)

    Returns:
        True if the index exists on the endpoint, False otherwise

    Raises:
        Exception: If an unexpected error occurs that isn't related to the index
                  not existing (e.g., permission issues)
    """
    try:
        # Attempt to describe the index - this will succeed only if the index exists
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        # Check if this is a "not exists" error or something else
        # Handle both "RESOURCE_DOES_NOT_EXIST" and "does not exist" error patterns
        error_str = str(e).lower()
        if (
            "does not exist" not in error_str
            and "resource_does_not_exist" not in error_str
        ):
            # For unexpected errors, provide a more helpful message
            print(
                "Unexpected error describing the index. This could be a permission issue."
            )
            raise e
    # If we reach here, the index doesn't exist
    return False


def find_index(
    vsc: VectorSearchClient, index_full_name: str
) -> tuple[bool, str | None]:
    """
    Find a Vector Search index across all endpoints.

    Searches all available endpoints to find where the index is located.

    Args:
        vsc: Databricks Vector Search client instance
        index_full_name: Fully qualified name of the index (catalog.schema.index)

    Returns:
        Tuple of (exists: bool, endpoint_name: str | None)
        - (True, endpoint_name) if index is found
        - (False, None) if index is not found on any endpoint
    """
    try:
        endpoints = vsc.list_endpoints().get("endpoints", [])
    except Exception as e:
        if "REQUEST_LIMIT_EXCEEDED" in str(e):
            print("WARN: couldn't list endpoints due to REQUEST_LIMIT_EXCEEDED error.")
            return (False, None)
        raise e

    for endpoint in endpoints:
        endpoint_name: str = endpoint["name"]
        try:
            vsc.get_index(endpoint_name, index_full_name).describe()
            return (True, endpoint_name)
        except Exception:
            # Index not on this endpoint, try next
            # Catches both "does not exist" and "RESOURCE_DOES_NOT_EXIST" errors,
            # as well as other errors (permission issues, etc.) - we continue
            # searching other endpoints regardless of error type
            continue

    return (False, None)
