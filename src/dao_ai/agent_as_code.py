import sys

import mlflow
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from mlflow.models import ModelConfig
from mlflow.pyfunc import ChatModel

from dao_ai.config import AppConfig
from dao_ai.graph import create_dao_ai_graph
from dao_ai.models import create_agent

mlflow.langchain.autolog()

model_config: ModelConfig = ModelConfig()
config: AppConfig = AppConfig(**model_config.to_dict())

log_level: str = config.app.log_level

logger.remove()
logger.add(sys.stderr, level=log_level)

graph: CompiledStateGraph = create_dao_ai_graph(config=config)

app: ChatModel = create_agent(graph)

mlflow.models.set_model(app)
