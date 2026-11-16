import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_data():
    np.random.seed(42)
    num_customers = 20500
    num_orders = 160000
    
    print("Generating Customers...")
    signup_start = datetime(2023, 1, 1)
    signup_end = datetime(2025, 12, 31)
    date_range = (signup_end - signup_start).days
    
    signup_dates = [signup_start + timedelta(days=int(np.random.randint(0, date_range))) for _ in range(num_customers)]
    genders = np.random.choice(['F', 'M', 'Other'], size=num_customers, p=[0.48, 0.49, 0.03])
    ages = np.clip(np.random.normal(loc=35, scale=10, size=num_customers).astype(int), 18, 75)
    
    countries = ['United States', 'United Kingdom', 'Germany', 'Canada', 'Australia', 'France']
    country_probs = [0.45, 0.20, 0.15, 0.10, 0.05, 0.05]
    customer_countries = np.random.choice(countries, size=num_customers, p=country_probs)
    
    city_dict = {
        'United States': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'],
        'United Kingdom': ['London', 'Manchester', 'Birmingham', 'Glasgow'],
        'Germany': ['Berlin', 'Munich', 'Frankfurt', 'Hamburg'],
        'Canada': ['Toronto', 'Vancouver', 'Montreal', 'Calgary'],
        'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth'],
        'France': ['Paris', 'Lyon', 'Marseille', 'Nice']
    }
    customer_cities = [np.random.choice(city_dict[c]) for c in customer_countries]
    channels = ['Paid Search', 'Organic Search', 'Social Media', 'Email', 'Affiliates', 'Direct']
    channel_probs = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]
    acquisition_channels = np.random.choice(channels, size=num_customers, p=channel_probs)
    
    customers_df = pd.DataFrame({
        'customer_id': [f"CUST_{i:06d}" for i in range(1, num_customers + 1)],
        'signup_date': signup_dates,
        'gender': genders,
        'age': ages,
        'country': customer_countries,
        'city': customer_cities,
        'acquisition_channel': acquisition_channels
    })
    
    print("Generating Orders...")
    customer_weights = np.random.pareto(a=1.8, size=num_customers)
    customer_weights /= customer_weights.sum()
    order_customer_indices = np.random.choice(num_customers, size=num_orders, p=customer_weights)
    
    product_categories = ['Electronics', 'Apparel', 'Home & Kitchen', 'Beauty & Personal Care', 'Sports', 'Books']
    category_probs = [0.25, 0.30, 0.20, 0.10, 0.10, 0.05]
    
    order_ids = [f"ORD_{i:08d}" for i in range(1, num_orders + 1)]
    order_customer_ids = customers_df['customer_id'].iloc[order_customer_indices].values
    
    order_dates = []
    for idx in order_customer_indices:
        s_date = customers_df['signup_date'].iloc[idx]
        max_days = (datetime(2026, 3, 31) - s_date).days
        if max_days < 1: max_days = 1
        order_dates.append(s_date + timedelta(days=np.random.randint(0, max_days)))
        
    categories = np.random.choice(product_categories, size=num_orders, p=category_probs)
    quantities = np.random.choice([1, 2, 3, 4, 5], size=num_orders, p=[0.65, 0.20, 0.10, 0.03, 0.02])
    
    unit_prices = []
    for cat in categories:
        if cat == 'Electronics': unit_prices.append(np.random.gamma(shape=5, scale=40))
        elif cat == 'Apparel': unit_prices.append(np.random.gamma(shape=3, scale=15))
        elif cat == 'Home & Kitchen': unit_prices.append(np.random.gamma(shape=4, scale=20))
        else: unit_prices.append(np.random.gamma(shape=2, scale=12))
    unit_prices = np.round(unit_prices, 2)
    
    discounts = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30], size=num_orders, p=[0.50, 0.20, 0.15, 0.08, 0.05, 0.02])
    revenue = np.round(quantities * unit_prices * (1 - discounts), 2)
    
    orders_df = pd.DataFrame({
        'order_id': order_ids, 'customer_id': order_customer_ids, 'order_date': order_dates,
        'product_category': categories, 'quantity': quantities, 'unit_price': unit_prices,
        'discount': discounts, 'revenue': revenue
    })
    
    print("Generating Sessions...")
    num_sessions = num_orders * 2
    session_indices = np.random.choice(num_customers, size=num_sessions, p=customer_weights)
    sessions_df = pd.DataFrame({
        'session_id': [f"SES_{i:08d}" for i in range(1, num_sessions + 1)],
        'customer_id': customers_df['customer_id'].iloc[session_indices].values,
        'session_date': [customers_df['signup_date'].iloc[idx] + timedelta(days=np.random.randint(0, max(1, (datetime(2026, 3, 31) - customers_df['signup_date'].iloc[idx]).days))) for idx in session_indices],
        'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=num_sessions, p=[0.60, 0.35, 0.05]),
        'session_duration': np.round(np.random.exponential(scale=180, size=num_sessions), 1),
        'pages_viewed': np.random.poisson(lam=4.5, size=num_sessions) + 1,
        'traffic_source': np.random.choice(channels, size=num_sessions, p=channel_probs)
    })
    
    print("Generating Support Tickets...")
    num_tickets = int(num_customers * 0.3)
    ticket_indices = np.random.choice(num_customers, size=num_tickets)
    support_df = pd.DataFrame({
        'ticket_id': [f"TCK_{i:06d}" for i in range(1, num_tickets + 1)],
        'customer_id': customers_df['customer_id'].iloc[ticket_indices].values,
        'ticket_date': [customers_df['signup_date'].iloc[idx] + timedelta(days=np.random.randint(0, max(1, (datetime(2026, 3, 31) - customers_df['signup_date'].iloc[idx]).days))) for idx in ticket_indices],
        'issue_type': np.random.choice(['Shipping Delay', 'Product Defect', 'Return/Refund', 'Payment Issue', 'General Inquiry'], size=num_tickets),
        'resolution_time_hours': np.round(np.random.gamma(shape=3, scale=8, size=num_tickets), 1),
        'satisfaction_score': np.random.choice([1, 2, 3, 4, 5], size=num_tickets, p=[0.08, 0.10, 0.15, 0.37, 0.30])
    })
    
    customers_df.to_csv("data/raw/customers.csv", index=False)
    orders_df.to_csv("data/raw/orders.csv", index=False)
    sessions_df.to_csv("data/raw/sessions.csv", index=False)
    support_df.to_csv("data/raw/customer_support.csv", index=False)
    print("Raw Datasets successfully generated inside data/raw/!")

if __name__ == "__main__":
    generate_data()