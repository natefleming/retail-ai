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
        if "RESOURCE_DOES_NOT_EXIST" not in str(e):
            # For unexpected errors, provide a more helpful message
            print(
                "Unexpected error describing the index. This could be a permission issue."
            )
            raise e
    # If we reach here, the index doesn't exist
    return False
