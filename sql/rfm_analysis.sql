-- RFM Analysis & Quintile Scoring
WITH base_metrics AS (
    SELECT
        customer_id,
        total_orders,
        total_revenue,
        days_since_last_purchase
    FROM customer_features
)
SELECT
    customer_id,
    NTILE(5) OVER (ORDER BY days_since_last_purchase ASC) AS r_score,
    NTILE(5) OVER (ORDER BY total_orders DESC) AS f_score,
    NTILE(5) OVER (ORDER BY total_revenue DESC) AS m_score
FROM base_metrics;