import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib

# 1. Load the 80% training dataset
df_train = pd.read_csv('churn-bigml-80.csv')
df_test = pd.read_csv('churn-bigml-20.csv')

# Combine them temporarily just to ensure identical encoding rules across sets
df_train['is_train'] = 1
df_test['is_train'] = 0
combined = pd.concat([df_train, df_test], axis=0)

# 2. Convert Target column (Churn) from True/False to 1/0
combined['Churn'] = combined['Churn'].astype(int)

# 3. Handle Categorical Columns
categorical_cols = ['State', 'International plan', 'Voice mail plan']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))
    encoders[col] = le

# Separate back into train and test sets
train_clean = combined[combined['is_train'] == 1].drop(columns=['is_train'])
test_clean = combined[combined['is_train'] == 0].drop(columns=['is_train'])

# 4. Separate Features and Target
X_train = train_clean.drop(columns=['Churn'])
y_train = train_clean['Churn']

X_test = test_clean.drop(columns=['Churn'])
y_test = test_clean['Churn']

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train high-performance XGBoost Classifier
model = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
model.fit(X_train_scaled, y_train)

# Print accuracy to verify performance
train_acc = model.score(X_train_scaled, y_train) * 100
test_acc = model.score(X_test_scaled, y_test) * 100
print(f"Model Training Complete!\nTrain Accuracy: {train_acc:.2f}%\nTest Accuracy: {test_acc:.2f}%")

# 7. Save production assets for your Streamlit App
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(X_train.columns.tolist(), 'feature_columns.pkl')
print("All model artifacts saved to disk successfully!")