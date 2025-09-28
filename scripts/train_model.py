import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Ensure output folder exists
os.makedirs('outputs/models', exist_ok=True)

# Load cleaned dataset (relative path from project root)
df = pd.read_csv('data/processed/customer_churn_cleaned.csv')

# Split features and target
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, 'outputs/models/random_forest_model.pkl')

print("Model training complete. Model saved in outputs/models/")
