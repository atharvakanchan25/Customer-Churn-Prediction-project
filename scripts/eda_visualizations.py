import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Ensure figures folder exists
os.makedirs('outputs/figures', exist_ok=True)

# Load cleaned dataset
df = pd.read_csv('data/processed/customer_churn_cleaned.csv')

# -----------------------------
# 1. Churn distribution plot
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.savefig('outputs/figures/churn_distribution.png')
plt.close()

# -----------------------------
# 2. Churn by Contract type
# -----------------------------
plt.figure(figsize=(8,5))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.savefig('outputs/figures/churn_by_contract.png')
plt.close()

# -----------------------------
# 3. Correlation heatmap
# -----------------------------
plt.figure(figsize=(12,10))
corr = df.drop('customerID', axis=1).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('outputs/figures/correlation_heatmap.png')
plt.close()

# -----------------------------
# 4. Feature importance from Random Forest
# -----------------------------
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_importances, y=feat_importances.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.savefig('outputs/figures/feature_importance.png')
plt.close()

print("EDA & feature importance plots saved in outputs/figures/")
