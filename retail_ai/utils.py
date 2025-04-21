from typing import Any, Callable

import importlib


def callable_from_fqn(fqn: str) -> Callable[[Any, ...], Any]:
  try:
    module_path, func_name = fqn.rsplit(".", 1)
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    if not callable(func):
      raise TypeError(f"Function {func_name} is not callable.")
    return func
  except(ImportError, AttributeError, TypeError) as e:
    raise ImportError(f"Failed to import {fqn}: {e}")