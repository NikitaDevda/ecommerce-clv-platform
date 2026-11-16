import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def train_models():
    print("1. Loading features and orders...")
    features = pd.read_parquet("data/processed/customer_features.parquet")
    orders = pd.read_parquet("data/processed/clean_orders.parquet")
    
    cutoff_date = pd.to_datetime("2026-03-31")
    target_end_date = cutoff_date + pd.Timedelta(days=180)
    
    future_orders = orders[(orders['order_date'] > cutoff_date) & (orders['order_date'] <= target_end_date)]
    target_revenue = future_orders.groupby('customer_id')['revenue'].sum().reset_index()
    target_revenue.rename(columns={'revenue': 'future_revenue_6m'}, inplace=True)
    
    ml_df = features.merge(target_revenue, on='customer_id', how='left')
    ml_df['future_revenue_6m'] = ml_df['future_revenue_6m'].fillna(0.0)
    
    cat_cols = ['gender', 'country', 'acquisition_channel']
    ml_df = pd.get_dummies(ml_df, columns=cat_cols, drop_first=True)
    
    drop_cols = ['customer_id', 'signup_date', 'city', 'first_order_date', 'last_order_date', 'future_revenue_6m']
    X = ml_df.drop(columns=[c for c in drop_cols if c in ml_df.columns])
    y = ml_df['future_revenue_6m']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    }
    
    best_model = None
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"-> {name} | RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f} | MAE: {mean_absolute_error(y_test, preds):.2f} | R2: {r2_score(y_test, preds):.4f}")
        best_model = model
        
    joblib.dump(best_model, "src/clv_model.pkl")
    print("Model saved successfully as src/clv_model.pkl")

if __name__ == "__main__":
    train_models()