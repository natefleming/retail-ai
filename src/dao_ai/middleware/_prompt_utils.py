"""Shared utilities for resolving prompt parameters in middleware."""

from __future__ import annotations

from dao_ai.config import PromptModel


def resolve_prompt(prompt: str | PromptModel, *, jinja: bool = False) -> str:
    """Resolve a prompt that may be a plain string or a :class:`PromptModel`.

    When a ``PromptModel`` is provided, its template is fetched from the
    prompt registry.  The *jinja* flag controls the template format:

    * ``jinja=False`` (default) – single-brace format (``{variable}``),
      suitable for ``str.format()`` and plain system prompts.
    * ``jinja=True`` – double-brace Jinja2 format (``{{ variable }}``),
      required by MLflow judges.

    Args:
        prompt: A raw prompt string **or** a ``PromptModel`` instance.
        jinja: If *True*, return the Jinja2-formatted template.

    Returns:
        The resolved prompt string.
    """
    if isinstance(prompt, PromptModel):
        return prompt.jinja_template if jinja else prompt.template
    return prompt
