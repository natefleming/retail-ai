from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HasFullName(ABC):
    @property
    @abstractmethod
    def full_name(self) -> str:
        pass


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


class SchemaModel(BaseModel, HasFullName):
    catalog_name: str
    schema_name: str
    permissions: list[PermissionModel]

    @property
    def full_name(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}"


class TableModel(BaseModel, HasFullName):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class LLMModel(BaseModel):
    name: str
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 8192


class VectorSearchEndpointType(str, Enum):
    STANDARD = "STANDARD"
    OPTIMIZED_STORAGE = "OPTIMIZED_STORAGE"


class VectorSearchEndpoint(BaseModel):
    name: str
    type: VectorSearchEndpointType


class IndexModel(BaseModel, HasFullName):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class VectorStoreModel(BaseModel):
    embedding_model: LLMModel
    endpoint: VectorSearchEndpoint
    index: IndexModel
    source_table: TableModel
    primary_key: str
    doc_uri: Optional[str] = None
    embedding_source_column: str
    columns: list[str]


class GenieRoomModel(BaseModel):
    name: str
    description: str
    space_id: str


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


class ConnectionModel(BaseModel, HasFullName):
    name: str

    @property
    def full_name(self) -> str:
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
    MCP = "mcp"


class BaseFunctionModel(BaseModel):
    type: FunctionType
    name: str


class PythonFunctionModel(HasFullName, BaseFunctionModel, BaseModel):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    type: FunctionType = FunctionType.PYTHON

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class FactoryFunctionModel(HasFullName, BaseFunctionModel, BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)
    type: FunctionType = FunctionType.FACTORY

    @property
    def full_name(self) -> str:
        return self.name


class McpFunctionModel(HasFullName, BaseFunctionModel, BaseModel):
    type: FunctionType = FunctionType.MCP

    @property
    def full_name(self) -> str:
        return self.name


class UnityCatalogFunctionModel(HasFullName, BaseFunctionModel, BaseModel):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    type: FunctionType = FunctionType.UNITY_CATALOG

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class ToolModel(BaseModel):
    name: str
    function: (
        PythonFunctionModel
        | FactoryFunctionModel
        | UnityCatalogFunctionModel
        | McpFunctionModel
    )


class GuardrailsModel(BaseModel):
    name: str
    model: LLMModel
    prompt: str


class CheckpointerTypeModel(str, Enum):
    POSTGRES = "postgres"


class CheckpointerModel(BaseModel):
    name: str
    type: CheckpointerTypeModel
    database: DatabaseModel


class AgentModel(BaseModel):
    name: str
    description: str
    model: LLMModel
    tools: list[ToolModel] = Field(default_factory=list)
    guardrails: list[GuardrailsModel] = Field(default_factory=list)
    checkpointer: Optional[CheckpointerModel] = None
    prompt: str
    handoff_prompt: Optional[str] = None


class SupervisorModel(BaseModel):
    model: LLMModel
    default_agent: AgentModel | str


class SwarmModel(BaseModel):
    model: LLMModel
    default_agent: AgentModel | str


class OrchestrationModel(BaseModel):
    supervisor: Optional[SupervisorModel] = None
    swarm: Optional[SwarmModel] = None

    @model_validator(mode="after")
    def validate_mutually_exclusive(self):
        if self.supervisor is not None and self.swarm is not None:
            raise ValueError("Cannot specify both supervisor and swarm")
        if self.supervisor is None and self.swarm is None:
            raise ValueError("Must specify either supervisor or swarm")
        return self


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


class LogLevel(str, Enum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class AppModel(BaseModel):
    log_level: LogLevel
    registered_model: RegisteredModelModel
    endpoint_name: str
    tags: dict[str, Any]
    permissions: list[AppPermissionModel]
    agents: list[AgentModel] = Field(default_factory=list)
    orchestration: OrchestrationModel


class EvaluationModel(BaseModel):
    model: LLMModel
    table: TableModel
    num_evals: int


class DatasetFormat(str, Enum):
    CSV = "csv"
    DELTA = "delta"
    JSON = "json"
    PARQUET = "parquet"
    ORC = "orc"


class DatasetModel(BaseModel):
    table: TableModel
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
    connections: dict[str, ConnectionModel] = Field(default_factory=dict)


class AppConfig(BaseModel):
    schemas: dict[str, SchemaModel]
    resources: ResourcesModel
    retrievers: dict[str, RetrieverModel] = Field(default_factory=dict)
    tools: dict[str, ToolModel] = Field(default_factory=dict)
    guardrails: dict[str, GuardrailsModel] = Field(default_factory=dict)
    checkpointer: Optional[CheckpointerModel] = None
    agents: dict[str, AgentModel] = Field(default_factory=dict)
    app: AppModel
    evaluation: Optional[EvaluationModel] = None
    datasets: Optional[list[DatasetModel]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    def find_agents(
        self, predicate: Callable[[AgentModel], bool] | None = None
    ) -> Sequence[AgentModel]:
        """
        Find agents in the configuration that match a given predicate.

        Args:
            predicate: A callable that takes an AgentModel and returns True if it matches.

        Returns:
            A list of AgentModel instances that match the predicate.
        """
        if predicate is None:

            def _null_predicate(agent: AgentModel) -> bool:
                return True

            predicate = _null_predicate

        return [agent for agent in self.agents.values() if predicate(agent)]

    def find_tools(
        self, predicate: Callable[[ToolModel], bool] | None = None
    ) -> Sequence[AgentModel]:
        """
        Find agents in the configuration that match a given predicate.

        Args:
            predicate: A callable that takes an AgentModel and returns True if it matches.

        Returns:
            A list of AgentModel instances that match the predicate.
        """
        if predicate is None:

            def _null_predicate(tool: ToolModel) -> bool:
                return True

            predicate = _null_predicate

        return [tool for tool in self.tools.values() if predicate(tool)]

    def find_guardrails(
        self, predicate: Callable[[GuardrailsModel], bool] | None = None
    ) -> Sequence[AgentModel]:
        """
        Find agents in the configuration that match a given predicate.

        Args:
            predicate: A callable that takes an AgentModel and returns True if it matches.

        Returns:
            A list of AgentModel instances that match the predicate.
        """
        if predicate is None:

            def _null_predicate(guardrails: GuardrailsModel) -> bool:
                return True

            predicate = _null_predicate

        return [
            guardrail for guardrail in self.guardrails.values() if predicate(guardrail)
        ]
