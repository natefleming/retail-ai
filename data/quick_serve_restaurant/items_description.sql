USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE items_description (
  item_name STRING,
  item_review STRING)
USING delta
COMMENT 'The items_description table in the coffeeshop retail database contains information about the various items available for purchase at the coffee shop, along with their corresponding reviews. This table serves as a reference for customers looking to learn more about specific items and their quality based on reviews. It helps the business track customer feedback and preferences, enabling them to make data-driven decisions on product offerings and improvements. The data in this table can be used for analyzing customer satisfaction, identifying popular items, and enhancing the overall shopping experience at the coffee shop.'
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'delta.enableDeletionVectors' = 'true',
  'delta.feature.changeDataFeed' = 'supported',
  'delta.feature.deletionVectors' = 'supported',
  'delta.feature.allowColumnDefaults' = 'supported',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7'
  )