-- Query 1: Customer spending
SELECT c.customer_id,
       c.city,
       COUNT(CASE WHEN o.status = 'completed' THEN 1 END) AS completed_orders,
       ROUND(COALESCE(SUM(CASE WHEN o.status = 'completed' THEN o.order_amount ELSE 0 END), 0), 2) AS total_spend_inr
FROM customers AS c
LEFT JOIN orders AS o ON o.customer_id = c.customer_id
GROUP BY c.customer_id, c.city
ORDER BY total_spend_inr DESC, c.customer_id;

-- Query 2: Revenue by city and category
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products AS p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Query 3: Above-average-priced products
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC, product_id;

-- Query 4: Customers without completed orders
SELECT c.customer_id, c.city
FROM customers AS c
WHERE NOT EXISTS (
    SELECT 1 FROM orders AS o
    WHERE o.customer_id = c.customer_id AND o.status = 'completed'
)
ORDER BY c.customer_id;

-- Query 5: Above-average customer spend using a CTE
WITH customer_spend AS (
    SELECT c.customer_id,
           c.city,
           COALESCE(SUM(CASE WHEN o.status = 'completed' THEN o.order_amount ELSE 0 END), 0) AS spend_inr
    FROM customers AS c
    LEFT JOIN orders AS o ON o.customer_id = c.customer_id
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, ROUND(spend_inr, 2) AS spend_inr
FROM customer_spend
WHERE spend_inr > (SELECT AVG(spend_inr) FROM customer_spend)
ORDER BY spend_inr DESC, customer_id;

-- Query 6: Customer spend rankings
WITH customer_spend AS (
    SELECT c.customer_id,
           c.city,
           COALESCE(SUM(CASE WHEN o.status = 'completed' THEN o.order_amount ELSE 0 END), 0) AS spend_inr
    FROM customers AS c
    LEFT JOIN orders AS o ON o.customer_id = c.customer_id
    GROUP BY c.customer_id, c.city
)
SELECT customer_id,
       city,
       ROUND(spend_inr, 2) AS spend_inr,
       RANK() OVER (ORDER BY spend_inr DESC) AS overall_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY spend_inr DESC) AS city_dense_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY spend_inr DESC, customer_id) AS city_row_number
FROM customer_spend
ORDER BY overall_rank, customer_id;

-- Query 7: Daily cumulative revenue
WITH daily_revenue AS (
    SELECT date(order_ts) AS order_date, SUM(order_amount) AS daily_revenue_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY date(order_ts)
)
SELECT order_date,
       ROUND(daily_revenue_inr, 2) AS daily_revenue_inr,
       ROUND(SUM(daily_revenue_inr) OVER (ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cumulative_revenue_inr
FROM daily_revenue
ORDER BY order_date;
