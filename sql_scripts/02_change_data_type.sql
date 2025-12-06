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
--------------------------------------------------------------------------


CREATE TABLE country_lookup (
    country_name VARCHAR(100) PRIMARY KEY,
    country_code VARCHAR(2) UNIQUE
);


INSERT INTO country_lookup (country_name, country_code) VALUES
('France', 'FR'),
('Philippines', 'PH'),
('United States', 'US'),
('Netherlands', 'NL'),
('Australia', 'AU'),
('Spain', 'ES'),
('United Kingdom', 'GB'),
('Germany', 'DE'),
('Canada', 'CA'),
('Brazil', 'BR')
ON CONFLICT (country_name) DO NOTHING;


ALTER TABLE public.events_staging
ADD COLUMN country_code VARCHAR(2);


UPDATE public.events_staging t1
SET country_code = t2.country_code
FROM country_lookup t2
WHERE t1.country = t2.country_name;


ALTER TABLE public.customers_staging
ADD COLUMN country_code VARCHAR(2);
UPDATE public.customers_staging t1
SET country_code = t2.country_code
FROM country_lookup t2
WHERE t1.country = t2.country_name;








