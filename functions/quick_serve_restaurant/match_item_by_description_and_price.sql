CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.match_item_by_description_and_price(
    description STRING,
    low_price DOUBLE default 0,
    high_price DOUBLE default 100,
    size STRING default 'Medium'
  )
  RETURNS TABLE(item_id STRING, item_name STRING, item_size STRING, category STRING, price DOUBLE)
  LANGUAGE SQL
  COMMENT 'This function finds coffee by matching the description, price range and returns relevant results. Example prompt: Suggest me some cold coffee options. or what is the price of a medium coffee?'
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
        inner join items_raw item
          ON vs.item_name = item.item_name
          and item.item_price BETWEEN low_price AND high_price
          and item.item_size ilike '%' || size || '%'
;