import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load raw datasets
customers = pd.read_csv('../data/raw/customers.csv')
orders = pd.read_csv('../data/raw/orders.csv')
sessions = pd.read_csv('../data/raw/sessions.csv')
support = pd.read_csv('../data/raw/customer_support.csv')

print('Customers shape:', customers.shape)
print('Orders shape:', orders.shape)
print('Sessions shape:', sessions.shape)
print('Support shape:', support.shape)

# Revenue distribution by product category
plt.figure(figsize=(10, 5))
sns.barplot(data=orders, x='product_category', y='revenue', estimator=sum)
plt.title('Total Revenue by Product Category')
plt.xticks(rotation=45)
plt.show()