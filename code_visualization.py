import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("bank_fraud_dataset.csv")

# Drop non-numeric columns before correlation
df_numeric = df.drop(columns=['Transaction_ID', 'Customer_ID', 'Timestamp', 'Location', 'Transaction_Type', 'Device_Used'])

# Ensure 'Fraudulent' is numeric
df_numeric['Fraudulent'] = df_numeric['Fraudulent'].astype(int)

# Heatmap of correlations (only numeric columns)
plt.figure(figsize=(12, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlations Heatmap")
plt.show()

# Fraud distribution across transaction types
plt.figure(figsize=(8, 4))
sns.countplot(x='Transaction_Type', hue='Fraudulent', data=df)
plt.xticks(rotation=45)
plt.title("Fraud Transactions by Type")
plt.show()
