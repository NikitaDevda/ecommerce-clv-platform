-- Customer Segmentation classification query
SELECT
    customer_segment,
    COUNT(customer_id) AS customer_count,
    SUM(total_revenue) AS segment_total_revenue,
    AVG(total_revenue) AS avg_revenue_per_customer
FROM customer_intelligence_master
GROUP BY 1
ORDER BY segment_total_revenue DESC;