import pandas as pd
import numpy as np

def compute_health_scores():
    print("Computing Customer Health Scores and RFM Segments...")
    features = pd.read_parquet("data/processed/customer_features.parquet")
    
    # RFM Scoring logic
    features['r_score'] = pd.qcut(features['days_since_last_purchase'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    features['f_score'] = pd.qcut(features['total_orders'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    features['m_score'] = pd.qcut(features['total_revenue'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    def segment_customer(row):
        r, f, m = row['r_score'], row['f_score'], row['m_score']
        if r >= 4 and f >= 4 and m >= 4: return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3: return 'Loyal Customers'
        elif r >= 3 and f <= 2: return 'Potential Loyalists'
        elif r <= 2 and f >= 3: return 'At Risk'
        else: return 'Lost Customers'
        
    features['customer_segment'] = features.apply(segment_customer, axis=1)
    
    def health_score(row):
        if row['days_since_last_purchase'] <= 30 and row['total_orders'] >= 3 and row['satisfaction_score'] >= 4:
            return 'Healthy'
        elif row['days_since_last_purchase'] <= 90 and row['total_orders'] >= 1:
            return 'Medium Risk'
        else:
            return 'High Risk'
            
    features['customer_health_status'] = features.apply(health_score, axis=1)
    
    features.to_parquet("data/processed/customer_intelligence_master.parquet", index=False)
    print("Customer Intelligence Master Table saved to data/processed/customer_intelligence_master.parquet")

if __name__ == "__main__":
    compute_health_scores()