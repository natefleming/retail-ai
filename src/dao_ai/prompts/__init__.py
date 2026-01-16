"""Prompt templates for DAO-AI components."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def get_prompt_path(name: str) -> Path:
    """Get the path to a prompt template file."""
    return PROMPTS_DIR / name
