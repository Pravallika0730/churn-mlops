import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib
import mlflow

# Paths
DATA_PATH = "data/processed/processed_churn.csv"
MODEL_PATH = "models/churn_model.pkl"

# Load processed data
df = pd.read_csv(DATA_PATH)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow experiment
mlflow.start_run()

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Log metric
mlflow.log_metric("roc_auc", roc_auc)

# Save model
joblib.dump(model, MODEL_PATH)
mlflow.log_artifact(MODEL_PATH)

mlflow.end_run()

print("Training completed.")
print("ROC-AUC:", roc_auc)
