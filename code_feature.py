import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("bank_fraud_dataset.csv")


# Frequency of transactions per customer
df['Transaction_Count'] = df.groupby('Customer_ID')['Customer_ID'].transform('count')

# If transaction amount is above 90th percentile, mark as high value
threshold = df['Transaction_Amount'].quantile(0.90)
df['High_Value_Transaction'] = (df['Transaction_Amount'] > threshold).astype(int)

print(df.head())

