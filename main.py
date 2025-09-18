# ================================
# Customer Churn Prediction - Telco Dataset
# ================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import openpyxl

# 1. Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Data Preprocessing
# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID (not useful for prediction)
customer_ids = df['customerID']
df.drop('customerID', axis=1, inplace=True)

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Split features & target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, customer_ids, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features
scaler = StandardScaler()
X_train[X_train.columns] = scaler.fit_transform(X_train)
X_test[X_test.columns] = scaler.transform(X_test)

# 3. Train Model
rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

# 4. Predictions
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# 5. Metrics
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "ROC AUC": roc_auc_score(y_test, y_proba)
}

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Threshold sweep (to find threshold for 80% precision)
thresholds = np.arange(0.1, 0.91, 0.1)
threshold_metrics = []
for t in thresholds:
    preds = (y_proba >= t).astype(int)
    threshold_metrics.append({
        "Threshold": round(t, 2),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds)
    })
threshold_df = pd.DataFrame(threshold_metrics)

# 6. Predictions DataFrame for Power BI
predictions_df = pd.DataFrame({
    "customerID": ids_test,
    "actual_churn": y_test.values,
    "predicted_churn": y_pred,
    "churn_probability": y_proba
})

# Add risk bucket
predictions_df["risk_bucket"] = pd.cut(
    predictions_df["churn_probability"],
    bins=[0, 0.33, 0.66, 1],
    labels=["Low", "Medium", "High"]
)

# Merge back important features for Power BI (tenure, contract, etc.)
predictions_df = predictions_df.merge(
    df.loc[ids_test.index][["tenure", "Contract", "PaymentMethod"]],
    left_index=True,
    right_index=True
)

# 7. Save results to Excel
with pd.ExcelWriter("churn_outputs.xlsx", engine="openpyxl") as writer:
    predictions_df.to_excel(writer, sheet_name="predictions", index=False)
    pd.DataFrame([metrics]).to_excel(writer, sheet_name="metrics", index=False)
    threshold_df.to_excel(writer, sheet_name="threshold_sweep", index=False)
    pd.DataFrame(cm).to_excel(writer, sheet_name="confusion_matrix", index=False)
    pd.DataFrame({"info": ["RandomForest (300 trees, max_depth=10, balanced)"]}).to_excel(writer, sheet_name="info", index=False)

print("✅ Processing done! Results saved in churn_outputs.xlsx")
