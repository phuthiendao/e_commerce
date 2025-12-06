-- customers_staging table
ALTER TABLE customers_staging
ADD CONSTRAINT pk_customers PRIMARY KEY (customer_id);


-- products_staging table
ALTER TABLE products_staging
ADD CONSTRAINT pk_products PRIMARY KEY (product_id);


-- events_staging table
ALTER TABLE events_staging
ADD CONSTRAINT pk_events PRIMARY KEY (event_id);

ALTER TABLE events_staging
ADD CONSTRAINT fk_events_customer
FOREIGN KEY (customer_id)
REFERENCES customers_staging (customer_id);


ALTER TABLE events_staging
ADD CONSTRAINT fk_events_product
FOREIGN KEY (product_id)
REFERENCES products_staging (product_id);








