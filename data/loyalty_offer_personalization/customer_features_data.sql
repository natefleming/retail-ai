USE IDENTIFIER(:database);

-- customer_features is built declaratively by the table DDL (CREATE OR REPLACE
-- TABLE AS ...). No additional seed rows needed — this file exists so the
-- dao-ai dataset pipeline can include customer_features as a regular dataset
-- entry without errors.
SELECT 1 AS noop;
