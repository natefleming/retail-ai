USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE faqs (
  faq_id BIGINT COMMENT 'Unique FAQ identifier' NOT NULL PRIMARY KEY
  ,category STRING COMMENT 'FAQ topic category (shipping, returns, account, products, b2b, payment)'
  ,question STRING COMMENT 'Customer-facing question'
  ,answer STRING COMMENT 'Authoritative answer (Vector Search embedding source via question || answer)'
  ,updated_at TIMESTAMP COMMENT 'Last reviewed timestamp'
)
CLUSTER BY AUTO
COMMENT 'Commerce Swarm FAQ knowledge base. Source table for faqs_vector_store.'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
;
