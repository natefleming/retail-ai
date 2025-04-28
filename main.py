import yaml
import mlflow
import json
from rich import print as pprint
from rich.console import Console

from retail_ai.config import AppConfig, Agent

def main():



    app_config: AppConfig = AppConfig(config="agent_as_config.yaml")

    globals()["__mlflow_model_config__"] = app_config.model_dump()
    app_config2: AppConfig = AppConfig()
    
    pprint(app_config.model_dump_json(indent=2))
    for agent_name, agent in app_config.agents.items():
        # Load the agent from the configuration
    
        
        # Print the loaded agent details
        print(f"Loaded agent: {agent.name}")
        print(f"Agent prompt: {agent.prompt}")
        print(f"Agent LLM: {agent.llm}")
        print(f"Agent tools: {agent.tools}")
        print(f"Agent checkpointer: {agent.checkpointer}")

    
if __name__ == "__main__":
    main()
