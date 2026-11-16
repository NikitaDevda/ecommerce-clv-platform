# Enterprise Customer Lifetime Value (CLV) & Customer Intelligence Platform

An end-to-end data science, analytics engineering, and business intelligence platform built for an e-commerce enterprise to predict 6-month future customer revenue, perform RFM segmentation, monitor customer health, and drive data-backed retention strategies.

---

## Business Objective
E-commerce businesses struggle with high customer acquisition costs (CAC) and uneven revenue distribution (Pareto principle). This platform solves three core business challenges:
1. **Revenue Forecasting:** Predict individual customer spend over the next 6 months.
2. **Behavioral Segmentation:** Group 20,500+ customers into actionable tiers (*Champions*, *Loyal*, *At Risk*, *Lost*).
3. **Churn Prevention:** Proactively flag *High-Risk* high-value accounts using engagement, recency, and support satisfaction metrics.

---

## Architecture & Tech Stack
* **Data Generation & ETL:** Python (`pandas`, `numpy`, `pyarrow`) simulating 20,500+ customers, 160,000+ orders, 320,000+ web sessions, and support tickets with realistic Pareto distributions and seasonality.
* **Data Warehousing & SQL:** Modular SQL scripts (`rfm_analysis.sql`, `customer_segments.sql`, `clv_metrics.sql`) using window functions (`NTILE`) and CTEs.
* **Machine Learning:** Regression pipeline comparing **Random Forest**, **XGBoost**, and **LightGBM** for 6-month future revenue prediction.
* **BI & Visualization:** Power BI Desktop data model (`final_powerbi_dataset.csv`) with custom DAX measures across 5 executive dashboard pages.

---

## Project Structure
```text
ecommerce-clv-platform/
├── data/
│   ├── raw/                      # Generated raw CSV datasets
│   └── processed/                # Cleaned parquet feature stores & Power BI master CSV
├── sql/
│   ├── rfm_analysis.sql          # Quintile RFM scoring queries
│   ├── customer_segments.sql     # Segment revenue aggregation
│   └── clv_metrics.sql           # Channel-level CLV metrics
├── src/
│   ├── dataset_generator.py      # Synthetic e-commerce data generator
│   ├── preprocess.py             # Data cleaning & schema standardization
│   ├── feature_engineering.py    # Customer-level feature store builder
│   ├── train_model.py            # Multi-model ML training & evaluation
│   ├── predict_clv.py            # Batch inference pipeline
│   └── customer_health_score.py  # RFM & Customer Health classification
├── dashboard/
│   └── powerbi_design.md         # 5-page Power BI wireframe & DAX specs
├── reports/
│   └── business_insights.md      # Strategic recommendations
├── requirements.txt
└── README.md