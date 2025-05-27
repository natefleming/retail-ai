from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any
from enum import Enum


class PrivilegeEnum(str, Enum):
    ALL_PRIVILEGES = "ALL_PRIVILEGES"
    USE_CATALOG = "USE_CATALOG"
    USE_SCHEMA = "USE_SCHEMA"
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    MODIFY = "MODIFY"
    CREATE = "CREATE"
    USAGE = "USAGE"
    CREATE_SCHEMA = "CREATE_SCHEMA"
    CREATE_TABLE = "CREATE_TABLE"
    CREATE_VIEW = "CREATE_VIEW"
    CREATE_FUNCTION = "CREATE_FUNCTION"
    CREATE_EXTERNAL_LOCATION = "CREATE_EXTERNAL_LOCATION"
    CREATE_STORAGE_CREDENTIAL = "CREATE_STORAGE_CREDENTIAL"
    CREATE_MATERIALIZED_VIEW = "CREATE_MATERIALIZED_VIEW"
    CREATE_TEMPORARY_FUNCTION = "CREATE_TEMPORARY_FUNCTION"
    EXECUTE = "EXECUTE"
    READ_FILES = "READ_FILES"
    WRITE_FILES = "WRITE_FILES"



class Permission(BaseModel):
    principals: list[str] = Field(default_factory=list)
    privileges: list[PrivilegeEnum]


class Schema(BaseModel):
    catalog_name: str
    schema_name: str
    full_name: str
    permissions: list[Permission]


class LLM(BaseModel):
    name: str
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 8192


class EmbeddingModel(BaseModel):
    name: str


class EndpointType(str, Enum):
    STANDARD = "STANDARD"


class Index(BaseModel):
    schema_model: Schema = Field(alias="schema")
    name: str


class SourceTable(BaseModel):
    schema_model: Schema = Field(alias="schema")
    name: str


class VectorStore(BaseModel):
    embedding_model: LLM
    endpoint_name: str
    endpoint_type: EndpointType
    index: Index
    source_table: SourceTable
    primary_key: str
    doc_uri: Optional[str] = None
    embedding_source_column: str
    columns: list[str]


class GenieRoom(BaseModel):
    name: str
    description: str
    space_id: str


class Table(BaseModel):
    schema_model: Schema = Field(alias="schema")
    name: str


class Volume(BaseModel):
    schema_model: Schema = Field(alias="schema")
    name: str


class Function(BaseModel):
    schema_model: Schema = Field(alias="schema")
    name: str


class Warehouse(BaseModel):
    name: str
    description: str
    warehouse_id: str


class Database(BaseModel):
    name: str
    connection_url: str
    connection_kwargs: dict[str, Any]


class SearchParameters(BaseModel):
    num_results: Optional[int] = 10
    filter: Optional[dict[str, Any]] = Field(default_factory=dict)
    query_type: Optional[str] = "ANN"


class Retriever(BaseModel):
    vector_store: VectorStore
    columns: list[str]
    search_parameters: SearchParameters


class FunctionType(str, Enum):
    PYTHON = "python"
    FACTORY = "factory"
    UNITY_CATALOG = "unity_catalog"


class PythonFunction(BaseModel):
    type: FunctionType
    schema_model: Schema = Field(alias="schema")
    name: str


class FactoryFunction(BaseModel):
    type: FunctionType
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class UnityCatalogFunction(BaseModel):
    type: FunctionType
    schema_model: Schema = Field(alias="schema")
    name: str


class Tool(BaseModel):
    name: str
    function: PythonFunction | FactoryFunction | UnityCatalogFunction


class Guardrail(BaseModel):
    model: LLM
    prompt: str


class CheckpointerType(str, Enum):
    POSTGRES = "postgres"


class Checkpointer(BaseModel):
    type: CheckpointerType
    database: Database


class Agent(BaseModel):
    name: str
    description: str
    model: LLM
    tools: list[Tool] = Field(default_factory=list)
    guardrails: list[Guardrail] = Field(default_factory=list)
    checkpointer: Optional[Checkpointer] = None
    prompt: str
    handoff_prompt: Optional[str] = None


class Supervisor(BaseModel):
    model: LLM
    default_agent: Agent | str


class Orchestration(BaseModel):
    supervisor: Supervisor


class RegisteredModel(BaseModel):
    schema_model: Schema = Field(alias="schema")
    name: str


class EntitlementEnum(str, Enum):
    CAN_MANAGE = "CAN_MANAGE"
    CAN_QUERY = "CAN_QUERY"
    CAN_VIEW = "CAN_VIEW"


class AppPermission(BaseModel):
    principals: list[str] = Field(default_factory=list)
    entitlements: list[EntitlementEnum]


class App(BaseModel):
    log_level: str
    registered_model: RegisteredModel
    endpoint_name: str
    tags: dict[str, Any]
    permissions: list[AppPermission]
    agents: list[Agent] = Field(default_factory=list)
    orchestration: Orchestration


class EvaluationTable(BaseModel):
    schema_model: Schema = Field(alias="schema")
    name: str


class Evaluation(BaseModel):
    model: LLM
    table: EvaluationTable
    num_evals: int


class DatasetFormat(str, Enum):
    PARQUET = "parquet"


class Dataset(BaseModel):
    schema_model: Schema = Field(alias="schema")
    table: str
    ddl: str
    data: str
    format: DatasetFormat


class Resources(BaseModel):
    llms: dict[str, LLM] = Field(default_factory=dict)
    vector_stores: dict[str, VectorStore] = Field(default_factory=dict)
    genie_rooms: dict[str, GenieRoom] = Field(default_factory=dict)
    tables: dict[str, Table] = Field(default_factory=dict)
    volumes: dict[str, Volume] = Field(default_factory=dict)
    functions: dict[str, Function] = Field(default_factory=dict)
    warehouses: dict[str, Warehouse] = Field(default_factory=dict)
    databases: dict[str, Database] = Field(default_factory=dict)


class AppConfig(BaseModel):
    schemas: dict[str, Schema]
    resources: Resources
    retrievers: dict[str, Retriever] = Field(default_factory=dict)
    tools: dict[str, Tool] = Field(default_factory=dict)
    guardrails: dict[str, Guardrail] = Field(default_factory=dict)
    checkpointer: Optional[Checkpointer] = None
    agents: dict[str, Agent] = Field(default_factory=dict) 
    app: App  
    evaluation: Optional[Evaluation] = None
    datasets: Optional[list[Dataset]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow", use_enum_values=True)


