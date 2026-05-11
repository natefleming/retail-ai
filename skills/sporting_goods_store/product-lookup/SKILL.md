---
name: product-lookup
description: Direct UPC/SKU product lookup. Returns a structured card with price, availability, and store location. Falls back to text search if UPC misses.
---

# Sporting Goods Product Lookup Skill

A focused skill for direct UPC/SKU lookups. Use this when the customer has
already identified a specific item and only needs structured information:
price, availability, store location, or product specs.

## Workflow

1. Extract the UPC or SKU from the customer's message.
2. Call the product lookup tool. If it returns no result, fall back to a
   text search using the product name.
3. Format the response as a structured card:

   ```
   Product: <name>
   UPC:     <upc>
   Price:   $<price>
   Stock:   <units> units (store: <store_id>)
   Specs:   <one-line summary>
   ```

4. If multiple stores have stock, list the closest two by store_id.

## Out of scope

If the customer wants comparison, recommendations, or general advice, hand
back to the main agent — that is a different skill.
