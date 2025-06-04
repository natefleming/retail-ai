import sys
import os

import mlflow
from mlflow.models import ModelConfig

from langgraph.graph.state import CompiledStateGraph
from mlflow.pyfunc import ChatModel
from retail_ai.graph import create_retail_ai_graph
from retail_ai.models import create_agent 
from retail_ai.config import AppConfig

from loguru import logger

mlflow.langchain.autolog()

model_config_path: str = os.getenv("MODEL_CONFIG_PATH", "../config/model_config.yaml")
model_config: ModelConfig = ModelConfig(development_config=model_config_path)
config: AppConfig = AppConfig(**model_config.to_dict())

log_level: str = config.app.log_level

logger.add(sys.stderr, level=log_level)

graph: CompiledStateGraph = create_retail_ai_graph(config=config)

app: ChatModel = create_agent(graph)

mlflow.models.set_model(app)
