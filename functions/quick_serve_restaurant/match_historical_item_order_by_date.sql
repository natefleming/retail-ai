CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.match_historical_item_order_by_date(
    description STRING,
    start_transaction_date STRING default current_timestamp(),
    end_transaction_date STRING default current_timestamp(),
    size STRING default 'Medium'
  )
  RETURNS TABLE(
    item_id STRING,
    item_name STRING,
    item_size STRING,
    category STRING,
    price DOUBLE,
    item_review STRING,
    total_order_value DOUBLE,
    in_or_out STRING,
    transaction_date TIMESTAMP
  )
  LANGUAGE SQL
  COMMENT 'This function finds coffee from by matching the description and transaction dates and returns arelevant results. Example scenario: Show me the history of coffee orders based on description and the dates of orders. or which was the most popular coffee?'
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
        inner join items_raw item
          ON vs.item_name = item.item_name
        INNER JOIN orders_raw orders
          ON orders.item_id = item.item_id
    where
      orders.created_at >= to_timestamp(start_transaction_date)
      and orders.created_at <= to_timestamp(end_transaction_date)
      and item.item_size ilike '%' || size || '%';