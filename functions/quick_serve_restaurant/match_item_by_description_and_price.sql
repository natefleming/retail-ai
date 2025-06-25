CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.match_item_by_description_and_price(
    description STRING COMMENT 'Coffee description or type to search for. Examples: "cold coffee", "espresso", "sweet drinks", "iced beverages", "strong coffee". Use customer preferences, drink types, or flavor profiles.',
    low_price DOUBLE default 0 COMMENT 'Minimum price filter in dollars. Examples: 0, 2.50, 3.00. Use when customers specify budget constraints like "under $5" or "at least $3". Defaults to $0 if not specified.',
    high_price DOUBLE default 100 COMMENT 'Maximum price filter in dollars. Examples: 5.00, 10.00, 15.00. Use when customers specify price limits like "under $6" or "no more than $8". Defaults to $100 if not specified.',
    size STRING default 'Medium' COMMENT 'Coffee size preference. Valid options: "Small", "Medium", "Large", or "N/A" for single-size items. Use when customers specify size preferences. Defaults to "Medium" if not specified.'
  )
  RETURNS TABLE(
    item_id STRING COMMENT 'Unique identifier for the coffee item',
    item_name STRING COMMENT 'Name of the coffee drink (e.g., "Cappuccino", "Cold Brew", "Caramel Latte")',
    item_size STRING COMMENT 'Available size for this item (Small, Medium, Large, or N/A)',
    category STRING COMMENT 'Coffee category or type (e.g., "Espresso", "Cold Brew", "Specialty", "Tea")',
    price DOUBLE COMMENT 'Price of the item in dollars for the specified size'
  )
  LANGUAGE SQL
  COMMENT 'Find coffee menu items by description and filter by price range and size. Use this tool when customers ask for recommendations, want to browse menu options, need items within a budget, or ask about pricing. Perfect for queries like "What cold drinks do you have?", "Show me coffee under $5", "What espresso drinks are available?", "I want something sweet and medium-sized", or "What are your cheapest drinks?"'
  RETURN
    SELECT
      item.item_id item_id,
      vs.item_name item_name,
      item.item_size item_size,
      item.item_cat category,
      item.item_price price
    FROM
      VECTOR_SEARCH(
        index => '{catalog_name}.{schema_name}.items_description_vs_index',
        query => description,
        num_results => 3
      ) vs
        inner join {catalog_name}.{schema_name}.items_raw item
          ON vs.item_name = item.item_name
          and item.item_price BETWEEN low_price AND high_price
          and item.item_size ilike '%' || size || '%'
;