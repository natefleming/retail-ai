import yaml
import mlflow
import json
from rich import print as pprint
from rich.console import Console

from retail_ai.config import AppConfig, Agent

def main():

    # Load the configuration file
    with open("agent_as_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    pprint(json.dumps(config, indent=2))

    app_config: AppConfig = AppConfig(**config)
    
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
