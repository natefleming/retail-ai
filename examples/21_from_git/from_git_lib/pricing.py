"""A tool *factory*, referenced as ``type: factory``.

The distinction from ``type: python`` matters for what this example proves: a
factory is called at agent-build time with the config's ``args:``, so a working
factory tool shows both that the module was importable from the checkout and
that the YAML arguments were threaded through to it.
"""

from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from loguru import logger


def create_discount_tool(
    discount_percent: float,
    minimum_price: float = 0.0,
) -> BaseTool:
    """Create a tool that applies the store's contractor discount to a price.

    Args:
        discount_percent: Percentage taken off the list price, e.g. ``12.5``.
        minimum_price: Prices at or below this value are not discounted.

    Returns:
        A LangChain tool that quotes the discounted price for a list price.
    """
    if not 0.0 <= discount_percent <= 100.0:
        raise ValueError(
            f"discount_percent must be between 0 and 100, got {discount_percent}"
        )

    logger.debug(
        "Creating discount tool",
        discount_percent=discount_percent,
        minimum_price=minimum_price,
    )

    @create_tool
    def apply_contractor_discount(list_price: float) -> str:
        """Apply the contractor discount to a product's list price.

        Use this whenever a customer asks what they would pay as a contractor,
        or asks for the discounted or "pro" price of an item. Pass the list
        price returned by a product lookup or search.

        Args:
            list_price: The product's list price in USD.

        Returns:
            A one-line quote showing the list price, the discount, and the
            discounted price.
        """
        if list_price <= minimum_price:
            return (
                f"${list_price:,.2f} is at or below the ${minimum_price:,.2f} "
                "discount floor, so the contractor price is the list price."
            )

        discounted: float = list_price * (1.0 - discount_percent / 100.0)
        return (
            f"List ${list_price:,.2f} - {discount_percent:g}% contractor "
            f"discount = ${discounted:,.2f}"
        )

    return apply_contractor_discount
