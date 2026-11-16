-- 6-Month CLV and Executive Performance Metrics
SELECT
    acquisition_channel,
    COUNT(customer_id) AS total_customers,
    AVG(total_revenue) AS avg_historical_revenue,
    AVG(predicted_future_revenue_6m) AS avg_predicted_clv_6m
FROM customer_intelligence_master
GROUP BY 1
ORDER BY avg_predicted_clv_6m DESC;