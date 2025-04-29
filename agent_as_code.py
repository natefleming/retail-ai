
from typing import Sequence
import sys

import mlflow
from mlflow.models import ModelConfig

from langchain_core.runnables import RunnableSequence
from langgraph.graph.state import CompiledStateGraph

from retail_ai.graph import create_graph
from retail_ai.models import LangGraphChatAgent, create_agent, as_langgraph_chain

from loguru import logger



mlflow.langchain.autolog()

config: ModelConfig = ModelConfig(development_config="model_config.yaml")

model_name: str = config.get("llms").get("model_name")

endpoint: str = config.get("retriever").get("endpoint_name")
index_name: str = config.get("retriever").get("index_name")
search_parameters: dict[str, str] = config.get("retriever").get("search_parameters")
primary_key: str = config.get("retriever").get("primary_key")
doc_uri: str = config.get("retriever").get("doc_uri")
text_column: str = config.get("retriever").get("embedding_source_column")
columns: Sequence[str] = config.get("retriever").get("columns")
space_id: str = config.get("genie").get("space_id")
log_level: str = config.get("app").get("log_level")

logger.add(sys.stderr, level=log_level)

graph: CompiledStateGraph = (
    create_graph(
        model_name=model_name,
        endpoint=endpoint,
        index_name=index_name,
        primary_key=primary_key,
        doc_uri=doc_uri,
        text_column=text_column,
        columns=columns,
        search_parameters=search_parameters,
        space_id=space_id
    )
)

#app: LangGraphChatAgent = create_agent(graph)

app: RunnableSequence = as_langgraph_chain(graph)

mlflow.models.set_model(app)
