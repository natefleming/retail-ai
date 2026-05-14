USE IDENTIFIER(:database);

-- One line per receipt for the first cut (item_count=1 implied per row here).
-- Brand is biased by customer cohort (customer_id % 10) so the same customer's
-- purchases cluster around their preferred brand — produces realistic
-- top_brands aggregations downstream.

INSERT INTO receipt_lines
WITH brand_lookup AS (
    SELECT * FROM VALUES
        (0, 'Nike',           'Footwear'),
        (1, 'Adidas',         'Footwear'),
        (2, 'Lululemon',      'Activewear'),
        (3, 'Patagonia',      'Outerwear'),
        (4, 'REI',            'Accessories'),
        (5, 'Levis',          'Denim'),
        (6, 'GAP',            'Apparel-Tops'),
        (7, 'JCrew',          'Apparel-Tops'),
        (8, 'BananaRepublic', 'Apparel-Bottoms'),
        (9, 'Puma',           'Footwear')
    AS t(cohort_idx, top_brand, top_category)
),
receipt_seed AS (
    SELECT
        receipt_id,
        customer_id,
        receipt_ts,
        cast(substring(customer_id, 3) AS INT) AS cust_num
    FROM receipts
)
SELECT
    concat('L-', lpad(cast(monotonically_increasing_id() AS STRING), 11, '0')) AS line_id,
    rs.receipt_id,
    -- SKU includes brand prefix so it's plausible
    concat(substring(b.top_brand, 1, 3), '-', lpad(cast(rand(501 + rs.cust_num) * 999 AS INT), 4, '0')) AS sku,
    -- 70% chance the line is from the customer's top brand, 30% random
    CASE WHEN rand(502 + rs.cust_num) < 0.70 THEN b.top_brand
         ELSE element_at(array('Nike','Adidas','Lululemon','Patagonia','REI','Levis','GAP','JCrew','BananaRepublic','Puma'),
                         cast(rand(503 + rs.cust_num) * 10 AS INT) + 1)
    END AS brand,
    CASE WHEN rand(504 + rs.cust_num) < 0.70 THEN b.top_category
         ELSE element_at(array('Footwear','Activewear','Outerwear','Denim','Apparel-Tops','Apparel-Bottoms','Accessories'),
                         cast(rand(505 + rs.cust_num) * 7 AS INT) + 1)
    END AS category,
    1 AS qty,
    round(rand(506) * 100 + 20, 2) AS line_price,
    CASE WHEN rand(507) < 0.30 THEN round(rand(508) * 15, 2) ELSE 0.0 END AS line_discount
FROM receipt_seed rs
JOIN brand_lookup b ON pmod(rs.cust_num, 10) = b.cohort_idx;
