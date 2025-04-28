import os
import pytest
import yaml
from pathlib import Path

from retail_ai.config import AppConfig, ToolType, CheckpointerType, GuardrailStrategy


@pytest.fixture
def sample_config():
    """Create a minimal valid configuration for testing"""
    return {
        "unity_catalog": {
            "catalog_name": "test_catalog",
            "database_name": "test_db"
        },
        "resources": {
            "llms": {
                "test_llm": {
                    "name": "Test LLM",
                    "model": "test-model"
                }
            }
        },
        "tools": {
            "test_tool": {
                "name": "Test Tool",
                "description": "A test tool",
                "function": {
                    "name": "test_function",
                    "type": "python",
                    "parameters": {}
                }
            }
        },
        "agents": {
            "test_agent": {
                "name": "Test Agent",
                "prompt": "You are a test agent",
                "llm": {"name": "Test LLM", "model": "test-model"},
                "tools": [
                    {"name": "Test Tool", "description": "A test tool", 
                     "function": {"name": "test_function", "type": "python", "parameters": {}}}
                ]
            }
        },
        "app": {
            "registered_model_name": "test_model",
            "endpoint_name": "test_endpoint",
            "agents": [
                {"name": "Test Agent", "prompt": "You are a test agent", 
                 "llm": {"name": "Test LLM", "model": "test-model"}, "tools": []}
            ]
        }
    }


@pytest.fixture
def sample_config_file(sample_config, tmp_path):
    """Create a temporary YAML config file"""
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)
    return config_file


def test_app_config_from_globals(sample_config):


    globals()["__mlflow_model_config__"] = sample_config
    app_config: AppConfig = AppConfig()
    assert app_config is not None
    
    
def test_app_config_init_from_dict(sample_config):
    """Test that AppConfig can be initialized from a dictionary"""
    app_config = AppConfig(config=sample_config)
    
    # Verify top-level fields are loaded correctly
    assert app_config.unity_catalog.catalog_name == "test_catalog"
    assert app_config.unity_catalog.database_name == "test_db"
    
    # Verify nested resources
    assert len(app_config.resources.llms) == 1
    assert app_config.resources.llms["test_llm"].name == "Test LLM"
    assert app_config.resources.llms["test_llm"].model == "test-model"
    
    # Verify tools
    assert len(app_config.tools) == 1
    assert app_config.tools["test_tool"].name == "Test Tool"
    assert app_config.tools["test_tool"].function.type == ToolType.PYTHON
    
    # Verify agents
    assert len(app_config.agents) == 1
    assert app_config.agents["test_agent"].name == "Test Agent"
    assert app_config.agents["test_agent"].prompt == "You are a test agent"
    
    # Verify app settings
    assert app_config.app.registered_model_name == "test_model"
    assert app_config.app.endpoint_name == "test_endpoint"


def test_load_config_from_file(sample_config_file):
    """Test that configuration can be loaded from a YAML file"""
    app_config = AppConfig(config=sample_config_file)
    
    # Verify basic config loading worked
    assert isinstance(app_config, AppConfig)
    assert app_config.unity_catalog.catalog_name == "test_catalog"
    assert app_config.unity_catalog.database_name == "test_db"
    

def test_load_config_with_guardrails(sample_config, tmp_path):
    """Test loading configuration with guardrails"""
    # Add guardrails to config
    sample_config["guardrails"] = {
        "test_guardrail": {
            "name": "Test Guardrail",
            "description": "A test guardrail",
            "strategy": "best_of_n",
            "evaluation_function": "test.evaluation",
            "reward_function": "test.reward",
            "N": 3,
            "threshold": 0.7,
            "failed_count": 2
        }
    }
    
    # Update an agent to use the guardrail
    sample_config["agents"]["test_agent"]["guardrails"] = [
        {
            "name": "Test Guardrail",
            "description": "A test guardrail",
            "strategy": "best_of_n",
            "evaluation_function": "test.evaluation",
            "reward_function": "test.reward",
            "N": 3,
            "threshold": 0.7,
            "failed_count": 2
        }
    ]
    
    config_file = tmp_path / "test_config_guardrails.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)
    
    app_config = AppConfig(config=config_file)
    
    # Verify guardrails are loaded
    assert len(app_config.guardrails) == 1
    assert app_config.guardrails["test_guardrail"].name == "Test Guardrail"
    assert app_config.guardrails["test_guardrail"].strategy == GuardrailStrategy.BEST_OF_N
    assert app_config.guardrails["test_guardrail"].N == 3
    assert app_config.guardrails["test_guardrail"].threshold == 0.7
    
    # Verify agent has guardrails
    assert len(app_config.agents["test_agent"].guardrails) == 1
    assert app_config.agents["test_agent"].guardrails[0].name == "Test Guardrail"


def test_load_config_with_agent_functions(sample_config, tmp_path):
    """Test loading configuration with agent functions"""
    # Add function to an agent
    sample_config["agents"]["test_agent"]["factory"] = {
        "name": "retail_ai.agents.test_function",
        "type": "python",
        "parameters": {"param1": "value1"}
    }
    
    config_file = tmp_path / "test_config_agent_functions.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)
    
    app_config = AppConfig(config=config_file)
    
    # Verify agent function is loaded
    assert app_config.agents["test_agent"].factory is not None
    assert app_config.agents["test_agent"].factory.name == "retail_ai.agents.test_function"
    assert app_config.agents["test_agent"].factory.type == ToolType.PYTHON
    assert app_config.agents["test_agent"].factory.parameters == {"param1": "value1"}


def test_load_real_config():
    """Test loading the actual configuration file from the project"""
    # Skip if the real config doesn't exist (CI environments)
    config_path = Path(__file__).parent.parent / "agent_as_config.yaml"
    if not config_path.exists():
        pytest.skip("agent_as_config.yaml not found")
    
    try:
        app_config = AppConfig(config=config_path)
        assert isinstance(app_config, AppConfig)
        # Basic validation that config loaded properly
        assert app_config.unity_catalog.catalog_name is not None
        assert app_config.unity_catalog.database_name is not None
        assert len(app_config.agents) > 0
    except Exception as e:
        pytest.fail(f"Failed to load actual config file: {e}")