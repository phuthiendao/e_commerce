SET search_path TO e_commerce_staging, public;

-------------------------------------------------------------------------------
-- 1. CREATE TABLE


-- 1.1 create countries table
CREATE TABLE e_commerce_staging.countries (
    country_code VARCHAR(3) PRIMARY KEY,
    country_name VARCHAR(100) NOT NULL ,
    region VARCHAR(50) NOT NULL ,
    latitude DECIMAL(10, 6),
    longitude DECIMAL(10, 6)
);


-- 1.2 create vendors table
CREATE TABLE e_commerce_staging.vendors (
    vendor_id SERIAL PRIMARY KEY,
    vendor_name VARCHAR(255) UNIQUE
);


-- 1.3 create resale_models table
CREATE TABLE e_commerce_staging.resale_models (
    resale_model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) UNIQUE
);


-- 1.4 create customers table
CREATE TABLE e_commerce_staging.customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    signup_date TIMESTAMP NOT NULL ,
    segment VARCHAR(50),
    acquisition_channel VARCHAR(50),
    age_band VARCHAR(20),
    country_code VARCHAR(3) NOT NULL ,
    currency_preference VARCHAR(3),
    FOREIGN KEY (country_code) REFERENCES countries(country_code)
);


--  1.5 create products table
CREATE TABLE e_commerce_staging.products (
    product_id VARCHAR(20) PRIMARY KEY,
    product_name VARCHAR(255),
    category VARCHAR(50),
    is_subscription BOOLEAN,
    billing_cycle VARCHAR(20),
    base_price_usd DECIMAL(10, 2),
    first_release_date DATE,
    vendor_id INT NOT NULL,
    resale_model_id INT NOT NULL,
    FOREIGN KEY (vendor_id) REFERENCES vendors(vendor_id),
    FOREIGN KEY (resale_model_id) REFERENCES resale_models(resale_model_id)
);


-- 1.6 create events table
CREATE TABLE e_commerce_staging.events (
    event_id VARCHAR(20) PRIMARY KEY,
    event_type VARCHAR(20),
    event_date TIMESTAMP,
    customer_id VARCHAR(20),
    product_id VARCHAR(20),
    country_code VARCHAR(3),
    channel VARCHAR(50),
    payment_method VARCHAR(50),
    currency VARCHAR(3),
    quantity INT NOT NULL CHECK (quantity > 0),
    unit_price_local DECIMAL(10, 2),
    discount_code VARCHAR(50),
    discount_local DECIMAL(10, 2),
    tax_local DECIMAL(10, 2),
    net_revenue_local DECIMAL(10, 2),
    fx_rate_to_usd DECIMAL(10, 6),
    net_revenue_usd DECIMAL(10, 2),
    is_refunded BOOLEAN,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (country_code) REFERENCES countries(country_code)

);


-- 1.7 create refunds table
CREATE TABLE e_commerce_staging.refunds (
    refund_id SERIAL PRIMARY KEY,
    event_id VARCHAR(20),
    refund_datetime TIMESTAMP,
    refund_reason VARCHAR(255),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);

----------------------------------------------------------------------------------------------------------------------
-- 2. INSERT VALUES INTO TABLE

-- countries table
INSERT INTO countries (country_code, country_name, region, latitude, longitude)
SELECT DISTINCT
    country_code AS country_code,
    country AS country_name,
    region,
    latitude,
    longitude
FROM public.events_staging
WHERE country IS NOT NULL
ON CONFLICT (country_code) DO NOTHING;


-- vendors table
INSERT INTO vendors (vendor_name)
SELECT DISTINCT vendor
FROM public.products_staging 
WHERE vendor IS NOT NULL
ON CONFLICT (vendor_name) DO NOTHING;


-- resale_models table
INSERT INTO resale_models (model_name)
SELECT DISTINCT resale_model
FROM public.products_staging
WHERE resale_model IS NOT NULL
ON CONFLICT (model_name) DO NOTHING;


-- customers table
INSERT INTO customers (
    customer_id, signup_date, segment, acquisition_channel,
    age_band, country_code, currency_preference
)
SELECT
    customer_id, signup_date, segment, acquisition_channel,
    age_band, country_code AS country_code, currency_preference
FROM public.customers_staging
WHERE customer_id NOT IN (SELECT customer_id FROM customers)
ON CONFLICT (customer_id) DO NOTHING;


-- products table
INSERT INTO products (
    product_id, product_name, category, is_subscription,
    billing_cycle, base_price_usd, first_release_date,
    vendor_id, resale_model_id
)
SELECT
    ps.product_id,
    ps.product_name,
    ps.category,
    ps.is_subscription,
    ps.billing_cycle,
    ps.base_price_usd,
    ps.first_release_date,
    v.vendor_id,
    rm.resale_model_id
FROM public.products_staging ps
LEFT JOIN vendors v ON ps.vendor = v.vendor_name
LEFT JOIN resale_models rm ON ps.resale_model = rm.model_name
WHERE ps.product_id NOT IN (SELECT product_id FROM products)
ON CONFLICT (product_id) DO NOTHING;


-- events table
INSERT INTO events (
    event_id, event_type, event_date, customer_id, product_id,
    country_code, channel, payment_method, currency, quantity,
    unit_price_local, discount_code, discount_local, tax_local,
    net_revenue_local, fx_rate_to_usd, net_revenue_usd, is_refunded
)
SELECT
    event_id, event_type, event_date, customer_id, product_id,
    country_code AS country_code, channel, payment_method, currency, quantity,
    unit_price_local, discount_code, discount_local, tax_local,
    net_revenue_local, fx_rate_to_usd, net_revenue_usd, is_refunded
FROM public.events_staging es 
WHERE event_id NOT IN (SELECT event_id FROM events)
ON CONFLICT (event_id) DO NOTHING;


-- refunds table
INSERT INTO refunds (event_id, refund_datetime, refund_reason)
SELECT
    event_id,
    refund_datetime,
    refund_reason
FROM public.events_staging
WHERE is_refunded = TRUE AND event_id NOT IN (SELECT event_id FROM refunds)
ON CONFLICT (refund_id) DO NOTHING;


-------------------------------------------------------------------------