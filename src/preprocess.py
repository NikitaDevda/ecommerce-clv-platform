import pandas as pd

def preprocess_data():
    print("Loading raw datasets...")
    customers = pd.read_csv("data/raw/customers.csv")
    orders = pd.read_csv("data/raw/orders.csv")
    sessions = pd.read_csv("data/raw/sessions.csv")
    support = pd.read_csv("data/raw/customer_support.csv")
    
    # Cleaning dates and handling formats
    customers['signup_date'] = pd.to_datetime(customers['signup_date'])
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    sessions['session_date'] = pd.to_datetime(sessions['session_date'])
    support['ticket_date'] = pd.to_datetime(support['ticket_date'])
    
    # Removing duplicates or invalid rows if any
    orders = orders.dropna(subset=['customer_id', 'order_id', 'revenue'])
    
    customers.to_parquet("data/processed/clean_customers.parquet", index=False)
    orders.to_parquet("data/processed/clean_orders.parquet", index=False)
    sessions.to_parquet("data/processed/clean_sessions.parquet", index=False)
    support.to_parquet("data/processed/clean_support.parquet", index=False)
    print("Preprocessing complete. Clean files saved to data/processed/")

if __name__ == "__main__":
    preprocess_data()