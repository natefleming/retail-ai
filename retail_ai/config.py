from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class UnityCatalogPrivilege(str, Enum):
    ALL_PRIVILEGES = "ALL_PRIVILEGES"
    MANAGE = "MANAGE"
    USE_CATALOG = "USE_CATALOG"
    USE_SCHEMA = "USE_SCHEMA"
    APPLY_TAG = "APPLY_TAG"
    BROWSE = "BROWSE"
    EXECUTE = "EXECUTE"
    READ_VOLUME = "READ_VOLUME"
    SELECT = "SELECT"
    MODIFY = "MODIFY"
    REFRESH = "REFRESH"
    WRITE_VOLUME = "WRITE_VOLUME"
    CREATE_FUNCTION = "CREATE_FUNCTION"
    CREATE_MATERIALIZED_VIEW = "CREATE_MATERIALIZED_VIEW"
    CREATE_MODEL = "CREATE_MODEL"
    CREATE_MODEL_VERSION = "CREATE_MODEL_VERSION"
    CREATE_SCHEMA = "CREATE_SCHEMA"
    CREATE_TABLE = "CREATE_TABLE"
    CREATE_VOLUME = "CREATE_VOLUME"
    DATABRICKS_STORAGE = "DATABRICKS_STORAGE"


class ToolType(str, Enum):
    UNITY_CATALOG = "unity_catalog"
    PYTHON = "python"


class CheckpointerType(str, Enum):
    POSTGRES = "postgres"
    MEMORY = "memory"


class GuardrailStrategy(str, Enum):
    BEST_OF_N = "best_of_n"
    REFINE = "refine"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Grant(BaseModel):
    principal: str
    privileges: List[UnityCatalogPrivilege]


class UnityCatalog(BaseModel):
    catalog_name: str
    database_name: str
    volume_name: Optional[str] = None
    grant: Optional[List[Grant]] = None


class LLM(BaseModel):
    name: str
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class Retriever(BaseModel):
    name: str
    description: str
    endpoint_name: str
    index_name: str
    embedding_model: Optional[Dict[str, Any]] = None
    endpoint_type: Optional[str] = None
    embedding_dimension: Optional[int] = None
    primary_key: Optional[str] = None
    doc_uri: Optional[str] = None
    embedding_source_column: Optional[str] = None
    columns_to_sync: Optional[List[str]] = None


class Function(BaseModel):
    name: str


class Warehouse(BaseModel):
    name: str
    description: str
    warehouse_id: str


class GenieRoom(BaseModel):
    name: str
    description: str
    space_id: str


class Database(BaseModel):
    name: str
    connection_url: str
    connection_kwargs: Optional[Dict[str, Any]] = None


class Connection(BaseModel):
    name: str
    description: str


class Resources(BaseModel):
    llms: Optional[Dict[str, LLM]] = None
    retrievers: Optional[Dict[str, Retriever]] = None
    functions: Optional[Dict[str, Function]] = None
    warehouses: Optional[Dict[str, Warehouse]] = None
    genie_rooms: Optional[Dict[str, GenieRoom]] = None
    databases: Optional[Dict[str, Database]] = None
    connections: Optional[Dict[str, Connection]] = None


class ToolFunction(BaseModel):
    name: Optional[str] = None
    type: ToolType
    parameters: Optional[Union[Dict[str, Any], str]] = None


class Tool(BaseModel):
    name: str
    description: str
    function: ToolFunction


class Checkpointer(BaseModel):
    type: CheckpointerType
    storage: Database


class Guardrail(BaseModel):
    name: str
    description: str
    strategy: GuardrailStrategy
    evaluation_function: str
    reward_function: str
    N: Optional[int] = None
    threshold: Optional[float] = None
    failed_count: Optional[int] = None


class AgentFunction(BaseModel):
    name: str
    type: ToolType
    parameters: Optional[Union[Dict[str, Any], str]] = {}


class Agent(BaseModel):
    name: str
    prompt: str
    handoff_prompt: Optional[str] = None
    llm: Union[LLM]
    tools: List[Tool]
    checkpointer: Optional[Checkpointer] = None
    guardrails: Optional[List[Guardrail]] = None
    function: Optional[AgentFunction] = None


class Message(BaseModel):
    role: MessageRole
    content: str


class InputExample(BaseModel):
    messages: List[Message]
    custom_inputs: Optional[Dict[str, Any]] = None
    configurable: Optional[Dict[str, Any]] = None


class App(BaseModel):
    registered_model_name: str
    endpoint_name: str
    tags: Optional[Dict[str, Any]] = None
    agents: List[Agent]
    input_examples: Optional[Dict[str, InputExample]] = None


class Evaluation(BaseModel):
    llm: LLM
    table_name: str
    num_evals: int


class HuggingFaceDataset(BaseModel):
    repo_id: str
    primary_key: str
    table_name: str


class AppConfig(BaseModel):
    """Main configuration model for Agent-as-Code deployments"""

    unity_catalog: UnityCatalog
    resources: Resources
    tools: Dict[str, Tool]
    checkpointer: Optional[Checkpointer] = None
    guardrails: Optional[Dict[str, Guardrail]] = None
    agents: Dict[str, Agent]
    app: App
    evaluation: Optional[Evaluation] = None


def load_config(file_path: str) -> AppConfig:
    """
    Load configuration from a YAML file

    Args:
        file_path: Path to the configuration YAML file

    Returns:
        Parsed configuration object
    """
    import yaml

    with open(file_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Convert unity-catalog key to unity_catalog for pydantic model
    if "unity_catalog" in config_data:
        config_data["unity_catalog"] = config_data.pop("unity_catalog")

    return AppConfig(**config_data)


def save_config(config: AppConfig, file_path: str) -> None:
    """
    Save configuration to a YAML file

    Args:
        config: Configuration object
        file_path: Path to save the configuration file
    """
    import yaml

    # Convert to dict and restore original key format
    config_dict = config.dict(exclude_none=True)
    if "unity_catalog" in config_dict:
        config_dict["unity_catalog"] = config_dict.pop("unity_catalog")

    with open(file_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
