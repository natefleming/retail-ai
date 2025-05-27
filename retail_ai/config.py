from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


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


class PermissionModel(BaseModel):
    principals: list[str] = Field(default_factory=list)
    privileges: list[PrivilegeEnum]


class SchemaModel(BaseModel):
    catalog_name: str
    schema_name: str
    full_name: str
    permissions: list[PermissionModel]


class LLMModel(BaseModel):
    name: str
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 8192


class EmbeddingModelModel(BaseModel):
    name: str


class EndpointType(str, Enum):
    STANDARD = "STANDARD"


class HasFullName(ABC):
    @property
    @abstractmethod
    def full_name(self) -> str:
        pass


class IndexModel(BaseModel, HasFullName):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class SourceTableModel(BaseModel, HasFullName):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class VectorStoreModel(BaseModel):
    embedding_model: LLMModel
    endpoint_name: str
    endpoint_type: EndpointType
    index: IndexModel
    source_table: SourceTableModel
    primary_key: str
    doc_uri: Optional[str] = None
    embedding_source_column: str
    columns: list[str]


class GenieRoomModel(BaseModel):
    name: str
    description: str
    space_id: str


class TableModel(BaseModel, HasFullName):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class VolumeModel(BaseModel, HasFullName):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class FunctionModel(BaseModel, HasFullName):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class WarehouseModel(BaseModel):
    name: str
    description: str
    warehouse_id: str


class DatabaseModel(BaseModel):
    name: str
    connection_url: str
    connection_kwargs: dict[str, Any]


class SearchParametersModel(BaseModel):
    num_results: Optional[int] = 10
    filter: Optional[dict[str, Any]] = Field(default_factory=dict)
    query_type: Optional[str] = "ANN"


class RetrieverModel(BaseModel):
    vector_store: VectorStoreModel
    columns: list[str]
    search_parameters: SearchParametersModel


class FunctionType(str, Enum):
    PYTHON = "python"
    FACTORY = "factory"
    UNITY_CATALOG = "unity_catalog"


class BaseFunctionModel(BaseModel):
    type: FunctionType
    name: str


class PythonFunctionModel(HasFullName, BaseFunctionModel, BaseModel):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class FactoryFunctionModel(HasFullName, BaseFunctionModel, BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return self.name


class UnityCatalogFunctionModel(HasFullName, BaseFunctionModel, BaseModel):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class ToolModel(BaseModel):
    name: str
    function: PythonFunctionModel | FactoryFunctionModel | UnityCatalogFunctionModel


class GuardrailModel(BaseModel):
    model: LLMModel
    prompt: str


class CheckpointerTypeModel(str, Enum):
    POSTGRES = "postgres"


class CheckpointerModel(BaseModel):
    type: CheckpointerTypeModel
    database: DatabaseModel


class AgentModel(BaseModel):
    name: str
    description: str
    model: LLMModel
    tools: list[ToolModel] = Field(default_factory=list)
    guardrails: list[GuardrailModel] = Field(default_factory=list)
    checkpointer: Optional[CheckpointerModel] = None
    prompt: str
    handoff_prompt: Optional[str] = None


class SupervisorModel(BaseModel):
    model: LLMModel
    default_agent: AgentModel | str


class OrchestrationModel(BaseModel):
    supervisor: SupervisorModel


class RegisteredModelModel(BaseModel, HasFullName):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class EntitlementEnum(str, Enum):
    CAN_MANAGE = "CAN_MANAGE"
    CAN_QUERY = "CAN_QUERY"
    CAN_VIEW = "CAN_VIEW"


class AppPermissionModel(BaseModel):
    principals: list[str] = Field(default_factory=list)
    entitlements: list[EntitlementEnum]


class AppModel(BaseModel):
    log_level: str
    registered_model: RegisteredModelModel
    endpoint_name: str
    tags: dict[str, Any]
    permissions: list[AppPermissionModel]
    agents: list[AgentModel] = Field(default_factory=list)
    orchestration: OrchestrationModel


class EvaluationTableModel(BaseModel):
    schema_model: SchemaModel = Field(alias="schema")
    name: str


class EvaluationModel(BaseModel):
    model: LLMModel
    table: EvaluationTableModel
    num_evals: int


class DatasetFormat(str, Enum):
    PARQUET = "parquet"


class DatasetModel(BaseModel):
    schema_model: SchemaModel = Field(alias="schema")
    table: str
    ddl: str
    data: str
    format: DatasetFormat


class ResourcesModel(BaseModel):
    llms: dict[str, LLMModel] = Field(default_factory=dict)
    vector_stores: dict[str, VectorStoreModel] = Field(default_factory=dict)
    genie_rooms: dict[str, GenieRoomModel] = Field(default_factory=dict)
    tables: dict[str, TableModel] = Field(default_factory=dict)
    volumes: dict[str, VolumeModel] = Field(default_factory=dict)
    functions: dict[str, FunctionModel] = Field(default_factory=dict)
    warehouses: dict[str, WarehouseModel] = Field(default_factory=dict)
    databases: dict[str, DatabaseModel] = Field(default_factory=dict)


class AppConfig(BaseModel):
    schemas: dict[str, SchemaModel]
    resources: ResourcesModel
    retrievers: dict[str, RetrieverModel] = Field(default_factory=dict)
    tools: dict[str, ToolModel] = Field(default_factory=dict)
    guardrails: dict[str, GuardrailModel] = Field(default_factory=dict)
    checkpointer: Optional[CheckpointerModel] = None
    agents: dict[str, AgentModel] = Field(default_factory=dict)
    app: AppModel
    evaluation: Optional[EvaluationModel] = None
    datasets: Optional[list[DatasetModel]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow", use_enum_values=True)
