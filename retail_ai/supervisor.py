from typing import Self, Sequence
from collections import OrderedDict
from retail_ai.config import AgentModel


class Supervisor:
    def __init__(self, agents: Sequence[AgentModel] = []) -> None:
        self.agent_registry: OrderedDict[str, AgentModel] = {}
        for agent in agents:
            self.register(agent)

    def register(self, agent: AgentModel) -> Self:
        self.agent_registry[agent.name] = agent

    @property
    def allowed_routes(self) -> Sequence[str]:
        agent_names: list[str] = list(self.agent_registry.keys())
        return sorted(agent_names)

    @property
    def prompt(self) -> str:
        prompt_result: str = "Analyze the user question and select ONE specific route from the allowed options:\n\n"

        for route in self.agent_registry:
            route: str
            handoff_prompt: str = self.agent_registry[route].handoff_prompt
            prompt_result += f"  - Route to '{route}': {handoff_prompt}\n"

        prompt_result += (
            "\n Choose exactly ONE route that BEST matches the user's primary intent."
        )

        return prompt_result
