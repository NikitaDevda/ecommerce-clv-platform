import pandas as pd
import joblib

def predict_future_clv():
    print("Loading model and feature store for inference...")
    features = pd.read_parquet("data/processed/customer_features.parquet")
    model = joblib.load("src/clv_model.pkl")
    
    inference_df = pd.get_dummies(features, columns=['gender', 'country', 'acquisition_channel'], drop_first=True)
    drop_cols = ['customer_id', 'signup_date', 'city', 'first_order_date', 'last_order_date']
    X_inf = inference_df.drop(columns=[c for c in drop_cols if c in inference_df.columns])
    
    predictions = model.predict(X_inf)
    
    output_df = pd.DataFrame({
        'customer_id': features['customer_id'],
        'predicted_future_revenue_6m': predictions
    })
    
    output_df.to_csv("data/processed/predicted_clv.csv", index=False)
    print("CLV predictions saved to data/processed/predicted_clv.csv")

if __name__ == "__main__":
    predict_future_clv()