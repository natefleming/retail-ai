"""A plain Python tool object, referenced as ``type: python``.

Deliberately dependency-free and deterministic: this tool exists to prove that
code from the git checkout reaches every runtime — the provisioning notebook, a
Model Serving endpoint, and a Databricks App — so it must not depend on Spark, a
warehouse, or network egress. It reports the module's own file path alongside
its answer, which makes the provenance assertable from a chat response.
"""

from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from loguru import logger

# Aisle and bin assignments by merchandise class, as they appear in the store
# planogram. Keys match `products.merchandise_class`.
_PLANOGRAM: dict[str, str] = {
    "POWER TOOLS": "Aisle 12, bins A1-A6",
    "HAND TOOLS": "Aisle 11, bins C2-C9",
    "PAINT": "Aisle 4, bins B1-B8",
    "FASTENERS": "Aisle 7, bins D1-D12",
    "PLUMBING": "Aisle 21, bins A3-A7",
}


@create_tool
def find_aisle(merchandise_class: str) -> str:
    """Find the aisle and bin range where a merchandise class is stocked.

    Use this after identifying a product's merchandise_class (for example
    "POWER TOOLS" or "PAINT") when the customer asks where to find an item in
    the store.

    Args:
        merchandise_class: Merchandise class exactly as it appears on the
            product record, e.g. "POWER TOOLS".

    Returns:
        A one-line location, plus the path this tool's module was loaded from.
    """
    key: str = merchandise_class.strip().upper()
    location: str | None = _PLANOGRAM.get(key)

    logger.debug(
        "Resolving planogram location",
        merchandise_class=key,
        found=location is not None,
    )

    if location is None:
        known: str = ", ".join(sorted(_PLANOGRAM))
        return f"No aisle on file for '{merchandise_class}'. Known classes: {known}."

    module_path: Path = Path(__file__).resolve()
    return f"{key} is stocked in {location}. [loaded from {module_path}]"


# `type: python` resolves a fully-qualified *tool object*, not a factory, so the
# decorated object is what the config points at.
find_aisle_tool: BaseTool = find_aisle
