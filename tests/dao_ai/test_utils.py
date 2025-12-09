import pytest

from dao_ai.utils import is_lib_provided, load_function


@pytest.mark.unit
def test_load_function_valid_builtin() -> None:
    """Test loading a valid built-in function."""
    func = load_function("builtins.len")
    assert callable(func)
    assert func([1, 2, 3]) == 3


@pytest.mark.unit
def test_load_function_valid_module_function() -> None:
    """Test loading a valid function from a standard library module."""
    func = load_function("os.path.join")
    assert callable(func)
    assert func("path", "to", "file") == "path/to/file"


@pytest.mark.unit
def test_load_function_invalid_module() -> None:
    """Test loading a function from a non-existent module."""
    with pytest.raises(ImportError, match="Failed to import nonexistent.module.func"):
        load_function("nonexistent.module.func")


@pytest.mark.unit
def test_load_function_invalid_function() -> None:
    """Test loading a non-existent function from a valid module."""
    with pytest.raises(ImportError, match="Failed to import os.nonexistent_function"):
        load_function("os.nonexistent_function")


@pytest.mark.unit
def test_load_function_non_callable() -> None:
    """Test loading a non-callable attribute."""
    with pytest.raises(ImportError, match="Failed to import os.name"):
        load_function("os.name")


@pytest.mark.unit
def test_load_function_langchain_tool() -> None:
    """Test loading a langchain tool (StructuredTool).

    In langchain 1.x, tools decorated with @tool return StructuredTool objects
    which are not directly callable (callable() returns False) but have an
    invoke() method.
    """
    tool = load_function("dao_ai.tools.current_time_tool")

    # Verify it's a langchain tool
    assert hasattr(tool, "invoke")
    assert hasattr(tool, "name")
    assert tool.name == "current_time_tool"

    # Verify it can be invoked
    result = tool.invoke({})
    assert isinstance(result, str)


@pytest.mark.unit
def test_load_function_no_dot_separator() -> None:
    """Test loading with invalid function name format."""
    with pytest.raises(ValueError):
        load_function("invalid_format")


@pytest.mark.unit
def test_is_lib_provided() -> None:
    """Test if a library is provided in the pip requirements."""
    assert is_lib_provided("dao-ai", ["dao-ai", "pandas"]) is True
    assert is_lib_provided("dao-ai", ["dao-ai>=0.0.1", "pandas"]) is True
    assert is_lib_provided("dao-ai", ["numpy", "pandas"]) is False
    assert (
        is_lib_provided(
            "dao-ai", ["git+https://github.com/natefleming/dao-ai.git", "numpy"]
        )
        is True
    )
