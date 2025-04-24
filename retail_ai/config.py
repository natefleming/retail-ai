from typing import Dict, List, Optional, Union, Literal, Any
from pydantic import BaseModel, Field


class UnityCatalogGrant(BaseModel):
    principal: str
    privileges: List[Literal[
        "ALL_PRIVILEGES", "MANAGE", "USE_CATALOG", "USE_SCHEMA", "APPLY_TAG",
        "BROWSE", "EXECUTE", "READ_VOLUME", "SELECT", "MODIFY", "REFRESH",
        "WRITE_VOLUME", "CREATE_FUNCTION", "CREATE_MATERIALIZED_VIEW",
        "CREATE_MODEL", "CREATE_MODEL_VERSION", "CREATE_SCHEMA", "CREATE_TABLE",
        "CREATE_VOLUME"
    ]]


class UnityCatalog(BaseModel):
    catalog_name: str
    database_name: str
    volume_name: str
    grant: Optional[List[UnityCatalogGrant]] = None


class LLM(BaseModel):
    name: str
    model: str
    temperature: Optional[float] = Field(default=None, ge=0, le=1)
    max_tokens: Optional[float] = Field(default=None, gt=0)


class Retriever(BaseModel):
    name: str
    description: str
    endpoint_name: str
    embedding_model: dict
    endpoint_type: Literal["STANDARD"]
    index_name: str
    embedding_dimension: int = Field(gt=0)
    primary_key: str
    doc_uri: str
    embedding_source_column: str
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


class DatabaseConnectionKwargs(BaseModel):
    autocommit: Optional[bool] = None
    prepare_threshold: Optional[int] = None


class Database(BaseModel):
    name: str
    connection_url: str
    connection_kwargs: DatabaseConnectionKwargs


class Resources(BaseModel):
    llms: Dict[str, LLM]
    retrievers: Dict[str, Retriever]
    functions: Dict[str, Function]
    warehouses: Dict[str, Warehouse]
    genie_rooms: Dict[str, GenieRoom]
    databases: Dict[str, Database]
    

class UnityCatalogFunctionDefinition(BaseModel):
    name: str
    type: Literal["unity-catalog"]


class PythonFunctionDefinition(BaseModel):
    name: str
    type: Literal["python"]
    parameters: dict


class Tool(BaseModel):
    name: str
    description: str
    function: Union[UnityCatalogFunctionDefinition, PythonFunctionDefinition]


class Checkpointer(BaseModel):
    type: Literal["postgres"]
    storage: Database


class Agent(BaseModel):
    name: str
    prompt: str
    handoff_prompt: str
    llm: LLM
    tools: List[Tool]
    checkpointer: Checkpointer


class Message(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: str


class ConfigurableInputs(BaseModel):
    thread_id: Optional[str] = None
    tone: Optional[str] = None


class InputExample(BaseModel):
    messages: List[Message]
    custom_inputs: Optional[dict] = None
    configurable: Optional[ConfigurableInputs] = None


class App(BaseModel):
    registered_model_name: str
    endpoint_name: str
    tags: Dict[str, Union[str, bool]]
    agents: List[Agent]
    input_examples: Dict[str, InputExample]


class Evaluation(BaseModel):
    llm: LLM
    table_name: str
    num_evals: int = Field(gt=0)


class HuggingFaceDataset(BaseModel):
    repo_id: str
    primary_key: str
    table_name: str


class Datasets(BaseModel):
    huggingface: Optional[HuggingFaceDataset] = None


class AppConfig(BaseModel):
    unity_catalog: UnityCatalog = Field(..., alias="unity-catalog")
    resources: Resources
    tools: Dict[str, Tool]
    checkpointer: Checkpointer
    agents: Dict[str, Agent]
    app: App
    evaluation: Optional[Evaluation] = None
    datasets: Optional[Datasets] = None