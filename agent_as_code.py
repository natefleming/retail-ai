
from typing import Sequence
import sys

import mlflow
from mlflow.models import ModelConfig

from langchain_core.runnables import RunnableSequence
from langgraph.graph.state import CompiledStateGraph

from retail_ai.graph import create_ace_arma_graph
from retail_ai.models import LangGraphChatAgent, create_agent, as_langgraph_chain

from loguru import logger


mlflow.langchain.autolog()

config: ModelConfig = ModelConfig(development_config="model_config.yaml")
log_level: str = config.get("app").get("log_level")

logger.add(sys.stderr, level=log_level)

graph: CompiledStateGraph = create_ace_arma_graph(model_config=config)

app: RunnableSequence = as_langgraph_chain(graph)

mlflow.models.set_model(app)
