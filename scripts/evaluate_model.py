import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Ensure output folder exists
os.makedirs('outputs/reports', exist_ok=True)

# Load cleaned dataset
df = pd.read_csv('data/processed/customer_churn_cleaned.csv')

# Split features and target
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
model = joblib.load('outputs/models/random_forest_model.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save report to file
report_path = 'outputs/reports/classification_report.txt'
with open(report_path, 'w') as f:
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
    f.write("\nAccuracy: " + str(accuracy_score(y_test, y_pred)))

print(f"Evaluation report saved in {report_path}")
