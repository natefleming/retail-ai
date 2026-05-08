-- Unity Catalog Function: find_nearby_stores_inventory
-- Description: Finds stores near a reference store that carry the requested SKUs in stock,
--              ordered by Haversine distance (great-circle miles).

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_nearby_stores_inventory(
  reference_store_id INT COMMENT 'Reference store identifier; distance is measured from this store'
  ,sku ARRAY<STRING> COMMENT 'One or more SKUs to check inventory for. Use another lookup tool first if you only have a product description.'
  ,max_results INT COMMENT 'Maximum number of nearby stores to return per SKU (e.g., 5)'
)
RETURNS TABLE(
  store_id INT COMMENT 'Identifier for the nearby store'
  ,store_name STRING COMMENT 'Display name of the nearby store'
  ,store_address STRING COMMENT 'Street address of the nearby store'
  ,store_city STRING COMMENT 'City of the nearby store'
  ,store_state STRING COMMENT 'State or province of the nearby store'
  ,store_phone STRING COMMENT 'Phone number for the nearby store'
  ,distance_miles DOUBLE COMMENT 'Great-circle distance from the reference store, in miles'
  ,sku STRING COMMENT 'SKU of the matched product'
  ,store_quantity INT COMMENT 'Available quantity at the nearby store'
  ,retail_amount DOUBLE COMMENT 'Current retail price at the nearby store'
)
READS SQL DATA
COMMENT 'Finds stores near a reference store that carry the requested SKUs in stock, ordered by Haversine distance.'
RETURN
WITH ref AS (
  SELECT latitude AS ref_lat, longitude AS ref_lon
  FROM {catalog_name}.{schema_name}.dim_stores
  WHERE store_id = find_nearby_stores_inventory.reference_store_id
),
candidate_stores AS (
  SELECT
    s.store_id, s.store_name, s.store_address, s.store_city, s.store_state, s.store_phone,
    3958.8 * 2 * ASIN(SQRT(
      POW(SIN(RADIANS(s.latitude - r.ref_lat) / 2), 2) +
      COS(RADIANS(r.ref_lat)) * COS(RADIANS(s.latitude)) *
      POW(SIN(RADIANS(s.longitude - r.ref_lon) / 2), 2)
    )) AS distance_miles
  FROM {catalog_name}.{schema_name}.dim_stores s
  CROSS JOIN ref r
  WHERE s.store_id != find_nearby_stores_inventory.reference_store_id
),
ranked AS (
  SELECT
    cs.store_id, cs.store_name, cs.store_address, cs.store_city, cs.store_state, cs.store_phone,
    cs.distance_miles,
    p.sku, i.store_quantity, i.retail_amount,
    DENSE_RANK() OVER (PARTITION BY p.sku ORDER BY cs.distance_miles ASC) AS distance_rank
  FROM candidate_stores cs
  JOIN {catalog_name}.{schema_name}.inventory i ON i.store_id = cs.store_id
  JOIN {catalog_name}.{schema_name}.products p ON p.product_id = i.product_id
  WHERE ARRAY_CONTAINS(find_nearby_stores_inventory.sku, p.sku)
    AND i.store_quantity > 0
)
SELECT store_id, store_name, store_address, store_city, store_state, store_phone,
       distance_miles, sku, store_quantity, retail_amount
FROM ranked
WHERE distance_rank <= find_nearby_stores_inventory.max_results
ORDER BY sku, distance_miles ASC;
