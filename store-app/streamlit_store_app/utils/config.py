"""Configuration utilities for the Streamlit Store App."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config() -> Dict[str, Any]:
    """Load application configuration."""
    config_path = Path(__file__).parent.parent / "config.yaml"

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    return config
