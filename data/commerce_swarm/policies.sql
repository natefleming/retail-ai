USE IDENTIFIER(:database);

CREATE TABLE IF NOT EXISTS policies (
  policy_id BIGINT COMMENT 'Unique policy identifier' NOT NULL PRIMARY KEY
  ,title STRING COMMENT 'Policy title'
  ,category STRING COMMENT 'Policy category (returns, shipping, privacy, b2b_terms, payment, safety)'
  ,body STRING COMMENT 'Full policy text — Vector Search embedding source'
  ,effective_date DATE COMMENT 'Date the policy became effective'
  ,updated_at TIMESTAMP COMMENT 'Last revision timestamp'
)
CLUSTER BY AUTO
COMMENT 'Commerce Swarm policy documents. Source table for policies_vector_store.'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
;
