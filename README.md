# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Step 1: Data Loading
# Load the dataset into a Pandas DataFrame
df = pd.read_csv('Online Retail Dataset.csv')

# Step 2: Exploratory Data Analysis (EDA)
# Explore the dataset
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Step 3: Data Cleaning
# Handle missing values
df = df.dropna()

# Step 4: Feature Engineering
# Convert 'InvoiceDate' to datetime format using a specific format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M')

# Extract date components
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day


# Use K-means clustering for customer segmentation
# Use K-means clustering for customer segmentation
X = df[['Quantity', 'UnitPrice']]
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
df['Segment'] = kmeans.labels_



# Analyze trends in transaction data over time
monthly_sales = df.groupby(['Year', 'Month'])['Quantity'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x=monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str), y=monthly_sales['Quantity'], marker='o')
plt.title('Monthly Sales Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()


# Identify top-selling products(quantity)
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
print(top_products)



