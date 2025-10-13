from typing import Any, Callable, Optional, Sequence

from langchain.prompts import PromptTemplate
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from loguru import logger

from dao_ai.config import PromptModel
from dao_ai.state import SharedState


def make_prompt(
    base_system_prompt: Optional[str | PromptModel],
) -> Callable[[dict, RunnableConfig], list]:
    logger.debug(f"make_prompt: {base_system_prompt}")

    def prompt(state: SharedState, config: RunnableConfig) -> list:
        system_prompt: str = ""
        if base_system_prompt:
            # Extract template string from PromptModel or use string directly
            template_str: str
            if isinstance(base_system_prompt, PromptModel):
                template_str = base_system_prompt.template
            else:
                template_str = base_system_prompt

            prompt_template: PromptTemplate = PromptTemplate.from_template(template_str)

            params: dict[str, Any] = {
                input_variable: "" for input_variable in prompt_template.input_variables
            }
            params |= config.get("configurable", {})

            system_prompt: str = prompt_template.format(**params)

        messages: Sequence[BaseMessage] = state["messages"]
        if system_prompt:
            messages = [SystemMessage(content=system_prompt)] + messages

        logger.debug(f"Created prompt with messages: {len(messages)}")
        logger.trace(f"Messages: {[m.model_dump() for m in messages]}")
        return messages

    return prompt
