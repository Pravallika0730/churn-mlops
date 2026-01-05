import joblib
import pandas as pd

# Load trained model
MODEL_PATH = "models/churn_model.pkl"
model = joblib.load(MODEL_PATH)

def predict_churn(input_data: dict):
    """
    input_data: dictionary of customer features
    returns: churn probability
    """
    df = pd.DataFrame([input_data])
    probability = model.predict_proba(df)[0][1]
    return probability
