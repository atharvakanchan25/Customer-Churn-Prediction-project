import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset (relative path)
df = pd.read_csv('data/raw/Telco-Customer-Churn.csv')
# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing values
df.dropna(inplace=True)

# Encode categorical columns
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                    'MultipleLines', 'InternetService', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                    'StreamingTV', 'StreamingMovies', 'Contract', 
                    'PaperlessBilling', 'PaymentMethod']

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Encode target
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Save cleaned dataset
df.to_csv('data/processed/customer_churn_cleaned.csv', index=False)
print("Preprocessing done. Cleaned data saved in data/processed/")
