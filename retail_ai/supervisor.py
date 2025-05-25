from typing import Any, Self, Sequence


class Supervisor:
    def __init__(self, agents: dict[str, Any] = {}) -> None:
        self.agents = agents

    def register(self, name: str, agent: dict[str, Any]) -> Self:
        self.agents[name] = agent
        return self

    @property
    def default_route(self) -> str | None:
        for name, agent in self.agents.items():
            if agent.get("is_default", False):
                return name
        return None

    @property
    def allowed_routes(self) -> Sequence[str]:
        agent_names: list[str] = list(self.agents.keys())
        return sorted(agent_names)

    @property
    def prompt(self) -> str:
        prompt_result: str = "Analyze the user question and select ONE specific route from the allowed options:\n\n"

        for route in self.allowed_routes:
            route: str
            handoff_prompt: str = self.agents[route].get("handoff_prompt", "")
            prompt_result += f"  - Route to '{route}': {handoff_prompt}\n"

        prompt_result += (
            "\n Choose exactly ONE route that BEST matches the user's primary intent."
        )

        return prompt_result
