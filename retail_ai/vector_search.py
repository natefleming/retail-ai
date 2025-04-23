
from databricks.vector_search.client import VectorSearchClient


def endpoint_exists(vsc: VectorSearchClient, vs_endpoint_name: str) -> bool:
    try:
        return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
    except Exception as e:
        if "REQUEST_LIMIT_EXCEEDED" in str(e):
            print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error.")
            return True
        else:
            raise e

def index_exists(vsc: VectorSearchClient, endpoint_name: str, index_full_name: str) -> bool:
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print('Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False