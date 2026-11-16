import pandas as pd
import numpy as np

def generate_features():
    print("Building Customer Feature Store...")
    customers = pd.read_parquet("data/processed/clean_customers.parquet")
    orders = pd.read_parquet("data/processed/clean_orders.parquet")
    sessions = pd.read_parquet("data/processed/clean_sessions.parquet")
    support = pd.read_parquet("data/processed/clean_support.parquet")
    
    snapshot_date = pd.to_datetime("2026-03-31")
    
    order_agg = orders.groupby('customer_id').agg(
        total_orders=('order_id', 'count'),
        total_revenue=('revenue', 'sum'),
        avg_order_value=('revenue', 'mean'),
        first_order_date=('order_date', 'min'),
        last_order_date=('order_date', 'max')
    ).reset_index()
    
    order_agg['days_since_last_purchase'] = (snapshot_date - order_agg['last_order_date']).dt.days
    total_months = max(1.0, (snapshot_date - orders['order_date'].min()).days / 30.0)
    order_agg['purchase_frequency'] = order_agg['total_orders'] / total_months
    
    session_agg = sessions.groupby('customer_id').agg(
        total_session_count=('session_id', 'count'),
        avg_session_duration=('session_duration', 'mean'),
        avg_pages_viewed=('pages_viewed', 'mean')
    ).reset_index()
    
    support_agg = support.groupby('customer_id').agg(
        customer_support_interaction_count=('ticket_id', 'count'),
        satisfaction_score=('satisfaction_score', 'mean')
    ).reset_index()
    
    features = customers.merge(order_agg, on='customer_id', how='left')
    features = features.merge(session_agg, on='customer_id', how='left')
    features = features.merge(support_agg, on='customer_id', how='left')
    
    features['total_orders'] = features['total_orders'].fillna(0)
    features['total_revenue'] = features['total_revenue'].fillna(0.0)
    features['avg_order_value'] = features['avg_order_value'].fillna(0.0)
    features['purchase_frequency'] = features['purchase_frequency'].fillna(0.0)
    features['days_since_last_purchase'] = features['days_since_last_purchase'].fillna(999)
    features['total_session_count'] = features['total_session_count'].fillna(0)
    features['avg_session_duration'] = features['avg_session_duration'].fillna(0.0)
    features['avg_pages_viewed'] = features['avg_pages_viewed'].fillna(0.0)
    features['customer_support_interaction_count'] = features['customer_support_interaction_count'].fillna(0)
    features['satisfaction_score'] = features['satisfaction_score'].fillna(3.0)
    
    features.to_parquet("data/processed/customer_features.parquet", index=False)
    print("Feature Engineering completed and saved to data/processed/customer_features.parquet")

if __name__ == "__main__":
    generate_features()