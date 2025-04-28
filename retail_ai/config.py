import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
from enum import Enum
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


class ToolFactory(BaseModel):
    name: Optional[str] = None
    type: Literal["python"] = None
    parameters: Optional[Union[Dict[str, Any], str]] = None
    
    
class Tool(BaseModel):
    name: str
    description: str
    function: Optional[ToolFunction] = None
    factory: Optional[ToolFactory] = None


class Checkpointer(BaseModel):
    type: CheckpointerType
    storage: Optional[Database] = None


class Guardrail(BaseModel):
    name: str
    description: str
    strategy: GuardrailStrategy
    evaluation_function: str
    reward_function: str
    N: Optional[int] = None
    threshold: Optional[float] = None
    failed_count: Optional[int] = None


class AgentFactory(BaseModel):
    name: str
    type: Optional[Literal["python"]] = None
    parameters: Optional[Union[Dict[str, Any], str]] = {}
    

class Agent(BaseModel):
    name: str
    prompt: str
    handoff_prompt: Optional[str] = None
    llm: Union[LLM]
    tools: List[Tool]
    checkpointer: Optional[Checkpointer] = None
    guardrails: Optional[List[Guardrail]] = None
    factory: Optional[AgentFactory] = None

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



class YamlIncludeLoader(yaml.SafeLoader):

    def __init__(self, stream):
        self._root = os.path.dirname(os.path.abspath(stream.name))
        super(YamlIncludeLoader, self).__init__(stream)
    
    @classmethod
    def include(cls, loader, node):
        """Process the !include directive"""
        filename = loader.construct_scalar(node)
        filepath = os.path.join(loader._root, filename)
        
        with open(filepath, 'r') as f:
            included_data = yaml.load(f, cls)
                        
        return included_data

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries recursively."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(file_path: str) -> AppConfig:

    YamlIncludeLoader.add_constructor("!include", YamlIncludeLoader.include)

    # Ensure we have an absolute path
    abs_file_path = os.path.abspath(file_path)

    try:
        with open(abs_file_path, "r") as f:
            # Directly load the file using the custom loader
            config = yaml.load(f, YamlIncludeLoader)
            
            # Check if there's an imported configuration to merge
            if "import" in config and isinstance(config["import"], dict):
                parent_config = config.pop("import")
                config = deep_merge(parent_config, config)
                print(f"Merged configuration with parent. Keys: {list(config.keys())}")
    except Exception as e:
        print(f"Error loading configuration from {abs_file_path}: {str(e)}")
        raise

    return AppConfig(**config)