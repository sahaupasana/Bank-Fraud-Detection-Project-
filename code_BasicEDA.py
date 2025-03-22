import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("bank_fraud_dataset.csv")

# Check first few rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Statistical summary
print(df.describe())

# Fraud vs. Non-Fraud Count
sns.countplot(x='Fraudulent', data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

# Transaction amount distribution
sns.boxplot(x='Fraudulent', y='Transaction_Amount', data=df)
plt.title("Transaction Amounts for Fraud vs Non-Fraud")
plt.show()