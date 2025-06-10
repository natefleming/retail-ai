# DatabricksProvider

The `DatabricksProvider` is a comprehensive service provider that automates the deployment of Databricks resources based on configuration files. It implements the `ServiceProviderBase` interface and provides methods for creating and managing various Databricks resources.

## Features

The `DatabricksProvider` can deploy the following resources:

- **Schemas**: Unity Catalog catalogs and schemas
- **Volumes**: Unity Catalog volumes for file storage
- **Vector Search**: Endpoints and indexes for vector similarity search
- **Functions**: Unity Catalog SQL functions
- **Datasets**: Tables populated from DDL and data files
- **Applications**: ML model serving endpoints with permissions

## Usage

### Basic Usage

```python
from retail_ai.providers.databricks import DatabricksProvider
from retail_ai.config import AppConfig
from mlflow.models import ModelConfig

# Load configuration
model_config = ModelConfig(development_config="config/model_config.yaml")
config = AppConfig(**model_config.to_dict())

# Initialize provider
provider = DatabricksProvider()

# Validate configuration
if provider.validate_config(config):
    # Deploy all resources
    provider.create_all_from_config(config)
```

### Individual Component Deployment

You can also deploy individual components:

```python
# Create schemas only
for schema_name, schema in config.schemas.items():
    provider.create_schema(schema)

# Create vector search resources only
for vs_name, vector_store in config.resources.vector_stores.items():
    provider.create_vector_search_endpoint(vector_store)
    provider.create_vector_search_index(vector_store)

# Create Unity Catalog functions only
for function in config.unity_catalog_functions:
    provider.create_function(function)
```

### Using the CLI

The provider can also be used through the command-line interface:

```bash
# Deploy all resources
retail-ai deploy

# Deploy with a specific configuration file
retail-ai deploy -c config/prod.yaml

# Validate configuration only
retail-ai deploy --validate-only

# Deploy only specific components
retail-ai deploy --component schemas
retail-ai deploy --component vector-search
retail-ai deploy --component functions
retail-ai deploy --component app

# Enable verbose logging
retail-ai deploy -vvv
```

## Methods

### Core Deployment Methods

#### `create_schema(schema: SchemaModel)`
Creates a Unity Catalog schema (catalog and database).

#### `create_volume(volume: VolumeModel, catalog_info: CatalogInfo, schema_info: SchemaInfo)`
Creates a Unity Catalog volume for file storage.

#### `create_vector_search_endpoint(vector_store: VectorStoreModel)`
Creates a vector search endpoint if it doesn't exist.

#### `create_vector_search_index(vector_store: VectorStoreModel)`
Creates a vector search index or syncs an existing one.

#### `create_function(function: UnityCatalogFunctionSqlModel)`
Creates a Unity Catalog function from SQL DDL and optionally tests it.

#### `create_dataset(dataset: DatasetModel)`
Creates a dataset table from DDL and data files.

#### `create_app(app: AppModel)`
Deploys an application using the Databricks agents framework and sets permissions.

### Utility Methods

#### `create_all_from_config(config: AppConfig)`
Orchestrates the deployment of all resources defined in an AppConfig in the correct order:
1. Schemas
2. Volumes
3. Datasets
4. Vector search resources
5. Unity Catalog functions
6. Applications

#### `validate_config(config: AppConfig) -> bool`
Validates that the configuration has all required resources and files exist.

## Configuration Requirements

The provider expects an `AppConfig` object with the following structure:

```yaml
schemas:
  my_schema:
    catalog_name: "my_catalog"
    schema_name: "my_schema"
    permissions: []

resources:
  vector_stores:
    my_vs:
      endpoint:
        name: "my_endpoint"
        type: "STANDARD"
      index:
        name: "my_index"
        schema: *my_schema
      source_table:
        name: "my_table"
        schema: *my_schema
      # ... other vector store config

  volumes:
    my_volume:
      name: "my_volume"
      schema: *my_schema

unity_catalog_functions:
  - function:
      name: "my_function"
      schema: *my_schema
    ddl: "functions/my_function.sql"
    test:
      parameters:
        param1: "value1"

datasets:
  - table:
      name: "my_table"
      schema: *my_schema
    ddl: "data/my_table.sql"
    data: "data/my_table.parquet"
    format: "parquet"

app:
  endpoint_name: "my_app"
  registered_model:
    name: "my_model"
    schema: *my_schema
  # ... other app config
```

## Error Handling

The provider includes comprehensive error handling:

- **File Validation**: Checks that DDL and data files exist before attempting to use them
- **Resource Validation**: Validates configuration completeness before deployment
- **Exception Handling**: Wraps operations in try-catch blocks with descriptive error messages
- **Graceful Degradation**: Continues deployment where possible even if individual components fail

## Dependencies

The provider requires the following Databricks SDK and MLflow dependencies:

- `databricks-sdk`
- `databricks-vector-search`
- `databricks-agents`
- `mlflow`
- `unitycatalog-ai[databricks]`
- `pyspark` (for dataset operations)

## Example Configurations

See the `config/` directory for example configuration files:

- `model_config.yaml` - Basic development configuration
- `model_config_dais.yaml` - Conference demo configuration
- `model_config_supervisor.yaml` - Supervisor orchestration pattern
- `model_config_swarm.yaml` - Swarm orchestration pattern

## Testing

Run the test suite to verify provider functionality:

```bash
# Run all provider tests
pytest tests/test_databricks_provider.py

# Run with verbose output
pytest tests/test_databricks_provider.py -v

# Run specific test
pytest tests/test_databricks_provider.py::TestDatabricksProvider::test_create_schema
```

## Contributing

When adding new functionality to the provider:

1. Add the method to the base `ServiceProviderBase` class if it should be part of the interface
2. Implement the method in `DatabricksProvider` with proper error handling
3. Add corresponding CLI support in `cli.py` if applicable
4. Write tests in `test_databricks_provider.py`
5. Update this documentation

## Troubleshooting

### Common Issues

1. **Authentication**: Ensure you have valid Databricks credentials configured
2. **Permissions**: Verify you have sufficient permissions to create resources
3. **File Paths**: Check that DDL and data file paths are correct and accessible
4. **Dependencies**: Ensure all required packages are installed

### Debug Mode

Enable verbose logging to see detailed operation logs:

```bash
retail-ai deploy -vvv  # DEBUG level logging
```

Or in Python:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
