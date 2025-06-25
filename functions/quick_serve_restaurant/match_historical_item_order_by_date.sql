CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.match_historical_item_order_by_date(
    description STRING COMMENT 'Coffee item description to search for in order history. Examples: "latte", "cold brew", "espresso", "frappuccino". Use customer-provided drink names or general categories.',
    start_transaction_date STRING default current_timestamp() COMMENT 'Start date for order history search in YYYY-MM-DD format or timestamp string. Examples: "2024-01-01", "2024-06-01 00:00:00". Defaults to current timestamp if not specified.',
    end_transaction_date STRING default current_timestamp() COMMENT 'End date for order history search in YYYY-MM-DD format or timestamp string. Examples: "2024-12-31", "2024-06-24 23:59:59". Defaults to current timestamp if not specified.',
    size STRING default 'Medium' COMMENT 'Coffee size filter for order history. Valid options: "Small", "Medium", "Large", or "N/A" for single-size items. Defaults to "Medium" if not specified.'
  )
  RETURNS TABLE(
    item_id STRING COMMENT 'Unique identifier for the coffee item in the order',
    item_name STRING COMMENT 'Name of the coffee item that was ordered',
    item_size STRING COMMENT 'Size of the coffee item (Small, Medium, Large, or N/A)',
    category STRING COMMENT 'Category or type of the coffee item (e.g., Espresso, Cold Brew, Specialty)',
    price DOUBLE COMMENT 'Unit price of the coffee item at time of order',
    item_review STRING COMMENT 'Customer review and description of the coffee item',
    total_order_value DOUBLE COMMENT 'Total value of this line item (price × quantity)',
    in_or_out STRING COMMENT 'Order type: whether it was for dine-in or takeout',
    transaction_date TIMESTAMP COMMENT 'Date and time when the order was placed'
  )
  LANGUAGE SQL
  COMMENT 'Retrieve historical coffee order data filtered by item description and date range. Use this tool when customers ask about past orders, order history, popular items over time, or want to reorder something they had before. Perfect for queries like "What did I order last week?", "Show my order history", "What was popular in January?", or "I want to reorder what I had yesterday".'
  RETURN
    SELECT
      item.item_id item_id,
      vs.item_name item_name,
      item.item_size item_size,
      item.item_cat category,
      item.item_price price,
      vs.item_review item_review,
      (item.item_price * orders.quantity) total_order_value,
      orders.in_or_out in_or_out,
      orders.created_at transaction_date
    FROM
      VECTOR_SEARCH(
        index => '{catalog_name}.{schema_name}.items_description_vs_index',
        query => description,
        num_results => 3
      ) vs
        inner join {catalog_name}.{schema_name}.items_raw item
          ON vs.item_name = item.item_name
        INNER JOIN {catalog_name}.{schema_name}.orders_raw orders
          ON orders.item_id = item.item_id
    where
      orders.created_at >= to_timestamp(start_transaction_date)
      and orders.created_at <= to_timestamp(end_transaction_date)
      and item.item_size ilike '%' || size || '%';