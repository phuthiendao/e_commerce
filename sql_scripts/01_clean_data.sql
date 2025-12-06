-- Backup table

-- Customers Table
CREATE TABLE customers_staging 
(LIKE customers);
INSERT INTO  customers_staging
SELECT *
FROM customers;


-- Events Table
CREATE TABLE events_staging 
(LIKE events);
INSERT INTO  events_staging
SELECT *
FROM events;


-- Products Table
CREATE TABLE products_staging 
(LIKE products);
INSERT INTO  products_staging
SELECT *
FROM products;

---------------------------------------------------------------------------------------------------------
-- Data cleaning
-- 1. Remove duplicated
-- 2. Standardize the data
-- 3. Null values or blank values
-- 4. Remove any columns

---------------------------------------------------------------------------------------------------------
-- customers_staging Table

-- Check duplicated row 
WITH duplicated_cte AS (
SELECT *,
row_number() OVER (PARTITION BY
customer_id,signup_date,region,currency_preference,segment,acquisition_channel,age_band,country) AS row_num
FROM customers_staging
)
SELECT *
FROM duplicated_cte 
WHERE row_num > 1


-- if necessary
-- Delete duplicate lines, keep 1 copy
--DELETE FROM customers_staging
--WHERE ctid IN (
--    SELECT ctid
--    FROM (
--        SELECT ctid,
--               ROW_NUMBER() OVER (
--                   PARTITION BY customer_id, signup_date, region, currency_preference, segment,
--                                acquisition_channel, age_band, country
--               ) AS row_num
--        FROM customers_staging
--    ) t
--    WHERE row_num > 1
--);


-- Check duplicated customer_id columns
SELECT customer_id, COUNT(*) AS cnt
FROM customers_staging
GROUP BY customer_id
HAVING COUNT(*) > 1;


-- if necessary
-- Delete duplicate customer_id, keep 1 copy
--DELETE FROM customers_staging
--WHERE ctid IN (
--    SELECT ctid
--    FROM (
--        SELECT ctid,
--               ROW_NUMBER() OVER (
--                   PARTITION BY customer_id
--                  ORDER BY signup_date DESC  -- Giữ bản mới nhất theo ngày đăng ký
--               ) AS row_num
--        FROM customers_staging
--    ) t
--    WHERE row_num > 1
--);


-- Check NULL in customer_id columns
SELECT *
FROM customers_staging
WHERE customer_id IS NULL;


-- 
--DELETE FROM customers_staging
--WHERE customer_id IS NULL;


-- Check NULL or Blanks values in region columns
SELECT country, region
FROM customers_staging
WHERE region IS NULL OR region ='';


SELECT country,region, COUNT(*) AS cnt
FROM customers_staging
GROUP BY country, region;


-- Updated region for the United States and Canada is NA (North America).
UPDATE customers_staging
SET region = 'NA'
WHERE country IN ('United States', 'Canada');


-- Standardize age_band
SELECT age_band
FROM customers_staging cs 
GROUP BY age_band;

UPDATE customers_staging
SET age_band = CASE
    WHEN age_band IN ('18-24', '25-34') THEN 'Young Adult (18-34)'
    WHEN age_band IN ('35-44', '45-54') THEN 'Middle Adult (35-54)'
    WHEN age_band = '55+' THEN 'Old Adult (55+)'
    ELSE age_band
END;

SELECT *
FROM customers_staging cs 

-----------------------------------------------------------------------------------------------
-- Products_staging Table

-- Check duplicated row 
WITH duplicated_cte AS (
SELECT *,
row_number() OVER (PARTITION BY
product_id, product_name, category, is_subscription, billing_cycle, vendor) AS row_num
FROM products_staging
)
SELECT *
FROM duplicated_cte 
WHERE row_num > 1


-- Check duplicated product_id columns
SELECT product_id, COUNT(*) AS cnt
FROM products_staging
GROUP BY product_id
HAVING COUNT(*) > 1;


-- Check NULL or Blank values in product_id columns
SELECT *	
FROM products_staging
WHERE product_id IS NULL OR product_id = ''

--------------------------------------------------------------------------------------

-- Events_staging Table

SELECT *
FROM events_staging es;

-- Check duplicated row 
WITH duplicated_cte AS (
SELECT *,
row_number() OVER (PARTITION BY event_id, event_type, customer_id, product_id, region, channel, payment_method, is_refunded
) AS row_num
FROM events_staging
)
SELECT *
FROM duplicated_cte 
WHERE row_num > 1

-- -- Delete duplicate row
WITH duplicated_cte AS (
    SELECT ctid, 
           row_number() OVER (
               PARTITION BY event_id, event_type, customer_id, product_id, region, channel, payment_method, is_refunded
               ORDER BY ctid
           ) AS row_num
    FROM events_staging
)
DELETE FROM events_staging e
USING duplicated_cte d
WHERE e.ctid = d.ctid
  AND d.row_num > 1;


-- Check duplicated event_id columns
SELECT event_id, COUNT(*) AS cnt
FROM events_staging
GROUP BY event_id
HAVING COUNT(*) > 1;

-- Check NULL values in event_id columns
SELECT COUNT(*) AS cnt
FROM events_staging
WHERE event_id IS NULL

-- Check Blank values in event_id columns
SELECT COUNT(*) AS cnt
FROM events_staging
WHERE event_id = '';

SELECT *
FROM events_staging es 
WHERE es.event_id = ''

-- Delete blank row
DELETE FROM events_staging
WHERE event_id = '';


-- Check NULL or Blanks values in region columns
SELECT country, region
FROM events_staging
WHERE region IS NULL OR region ='';

SELECT country,region, COUNT(*) AS cnt
FROM events_staging
GROUP BY country, region;

-- Updated region for the United States and Canada is NA (North America).
UPDATE events_staging
SET region = 'NA'
WHERE country IN ('United States', 'Canada');

SELECT *
FROM customers_staging cs 

--------------------------------------------------------------------------------------
-- Determine the data types of tables

-- customers_staging table
SELECT attname AS column_name, format_type(atttypid, atttypmod) AS data_type
FROM pg_attribute
WHERE attrelid = 'customers_staging'::regclass
  AND attnum > 0
  AND NOT attisdropped
ORDER BY attnum;


-- products_staging table
SELECT attname AS column_name, format_type(atttypid, atttypmod) AS data_type
FROM pg_attribute
WHERE attrelid = 'products_staging'::regclass
  AND attnum > 0
  AND NOT attisdropped
ORDER BY attnum;


-- events_staging table
SELECT attname AS column_name, format_type(atttypid, atttypmod) AS data_type
FROM pg_attribute
WHERE attrelid = 'events_staging'::regclass
  AND attnum > 0
  AND NOT attisdropped
ORDER BY attnum;


-- customers_staging table
ALTER TABLE customers_staging
  ALTER COLUMN customer_id TYPE TEXT USING NULLIF(customer_id, '')::TEXT,
  ALTER COLUMN signup_date TYPE TIMESTAMP USING NULLIF(signup_date, '')::TIMESTAMP,
  ALTER COLUMN region TYPE TEXT USING NULLIF(region, '')::TEXT,
  ALTER COLUMN currency_preference TYPE VARCHAR(3) USING NULLIF(currency_preference, '')::VARCHAR(3),
  ALTER COLUMN segment TYPE TEXT USING NULLIF(segment, '')::TEXT,
  ALTER COLUMN acquisition_channel TYPE TEXT USING NULLIF(acquisition_channel, '')::TEXT,
  ALTER COLUMN age_band TYPE TEXT USING NULLIF(age_band, '')::TEXT,
  ALTER COLUMN country TYPE TEXT USING NULLIF(country, '')::TEXT,
  ALTER COLUMN country_latitude TYPE DOUBLE PRECISION,
  ALTER COLUMN country_longitude TYPE DOUBLE PRECISION;


-- products_staging table
ALTER TABLE products_staging
  ALTER COLUMN product_id TYPE TEXT USING NULLIF(product_id, '')::TEXT,
  ALTER COLUMN product_name TYPE TEXT USING NULLIF(product_name, '')::TEXT,
  ALTER COLUMN category TYPE TEXT USING NULLIF(category, '')::TEXT,
  ALTER COLUMN billing_cycle TYPE TEXT USING NULLIF(billing_cycle, '')::TEXT,
  ALTER COLUMN vendor TYPE TEXT USING NULLIF(vendor, '')::TEXT,
  ALTER COLUMN resale_model TYPE TEXT USING NULLIF(resale_model, '')::TEXT,
  ALTER COLUMN brand_safe_name TYPE TEXT USING NULLIF(brand_safe_name, '')::TEXT,
  ALTER COLUMN product_name_orig TYPE TEXT USING NULLIF(product_name_orig, '')::TEXT,
  ALTER COLUMN base_key TYPE TEXT USING NULLIF(base_key, '')::TEXT,
  ALTER COLUMN product_version TYPE TEXT USING NULLIF(product_version, '')::TEXT,
  ALTER COLUMN base_price_usd TYPE DOUBLE PRECISION,
  ALTER COLUMN base_price_usd_orig TYPE DOUBLE PRECISION,
  ALTER COLUMN first_release_date TYPE TIMESTAMP USING 
    CASE
      WHEN first_release_date ~ '^\d{1,2}/\d{1,2}/\d{4}$' 
      THEN TO_TIMESTAMP(first_release_date, 'MM/DD/YYYY')
      ELSE NULL
    END;


ALTER TABLE events_staging
  ALTER COLUMN event_id TYPE TEXT USING NULLIF(event_id, '')::TEXT,
  ALTER COLUMN event_type TYPE TEXT USING NULLIF(event_type, '')::TEXT,
  ALTER COLUMN customer_id TYPE TEXT USING NULLIF(customer_id, '')::TEXT,
  ALTER COLUMN product_id TYPE TEXT USING NULLIF(product_id, '')::TEXT,
  ALTER COLUMN country TYPE TEXT USING NULLIF(country, '')::TEXT,
  ALTER COLUMN region TYPE TEXT USING NULLIF(region, '')::TEXT,
  ALTER COLUMN channel TYPE TEXT USING NULLIF(channel, '')::TEXT,
  ALTER COLUMN payment_method TYPE TEXT USING NULLIF(payment_method, '')::TEXT,
  ALTER COLUMN currency TYPE TEXT USING NULLIF(currency, '')::TEXT,
  ALTER COLUMN discount_code TYPE TEXT USING NULLIF(discount_code, '')::TEXT,
  ALTER COLUMN refund_reason TYPE TEXT USING NULLIF(refund_reason, '')::TEXT,
  ALTER COLUMN latitude TYPE DOUBLE PRECISION,
  ALTER COLUMN longitude TYPE DOUBLE PRECISION,
  ALTER COLUMN unit_price_local TYPE DOUBLE PRECISION,
  ALTER COLUMN discount_local TYPE DOUBLE PRECISION,
  ALTER COLUMN tax_local TYPE DOUBLE PRECISION,
  ALTER COLUMN net_revenue_local TYPE DOUBLE PRECISION,
  ALTER COLUMN fx_rate_to_usd TYPE DOUBLE PRECISION,
  ALTER COLUMN net_revenue_usd TYPE DOUBLE PRECISION,
  ALTER COLUMN event_date TYPE TIMESTAMP USING 
    CASE
      WHEN NULLIF(event_date,'') IS NOT NULL 
      THEN TO_TIMESTAMP(event_date, 'MM/DD/YYYY HH24:MI')
      ELSE NULL
    END,
  ALTER COLUMN refund_datetime TYPE TIMESTAMP USING 
    CASE
      WHEN NULLIF(refund_datetime,'') IS NOT NULL 
      THEN TO_TIMESTAMP(refund_datetime, 'MM/DD/YYYY HH24:MI')
      ELSE NULL
    END;



















