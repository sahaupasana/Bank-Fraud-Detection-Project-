import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("bank_fraud_dataset.csv")

# Drop non-numeric columns and encode categorical variables if necessary
df_numeric = df.drop(columns=['Transaction_ID', 'Customer_ID', 'Timestamp', 'Location', 'Device_Used'])

# Convert categorical 'Transaction_Type' to numeric using one-hot encoding
df_numeric = pd.get_dummies(df_numeric, columns=['Transaction_Type'], drop_first=True)

# Ensure 'Fraudulent' is numeric
df_numeric['Fraudulent'] = df_numeric['Fraudulent'].astype(int)

# Splitting data into features (X) and target variable (y)
X = df_numeric.drop(columns=['Fraudulent'])
y = df_numeric['Fraudulent']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
