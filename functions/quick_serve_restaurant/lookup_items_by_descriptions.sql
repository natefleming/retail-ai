CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.lookup_items_by_descriptions(
    description STRING
  )
  RETURNS TABLE(
    item_review STRING
  )
  LANGUAGE SQL
  COMMENT 'The items_description table in the coffeeshop retail database contains information about the various items available for purchase at the coffee shop, along with their corresponding reviews. This function serves as a reference for customers looking to learn more about specific items and their quality based on reviews. It helps the business track customer feedback and preferences, enabling them to make data-driven decisions on product offerings and improvements. The data in this table can be used for analyzing customer satisfaction, identifying popular items, and enhancing the overall shopping experience at the coffee shop.'
  RETURN 
    SELECT
      item_review item_review
    FROM
      VECTOR_SEARCH(
        index => '{catalog_name}.{schema_name}.items_description_vs_index',
        query => description,
        num_results => 1
      )