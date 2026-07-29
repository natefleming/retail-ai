CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.lookup_items_by_descriptions(
    description STRING COMMENT 'The name or description of a coffee item to look up. Examples: "cappuccino", "cold brew", "caramel macchiato", "iced latte". Use customer-provided drink names or descriptive terms.'
  )
  RETURNS TABLE(
    item_review STRING COMMENT 'Detailed customer review and description of the coffee item, including taste profile, ingredients, and customer feedback to help with recommendations'
  )
  LANGUAGE SQL
  COMMENT 'Look up detailed reviews and descriptions for coffee menu items. Use this tool when customers ask about specific drinks, want to know what something tastes like, need ingredient information, or want reviews before ordering. Returns customer reviews and detailed descriptions to help customers make informed choices.'
  RETURN 
    SELECT
      item_review item_review
    FROM
      VECTOR_SEARCH(
        index => '{catalog_name}.{schema_name}.items_description_vs_index',
        query => description,
        num_results => 1
      )