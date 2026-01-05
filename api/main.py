from fastapi import FastAPI
from inference.predict import predict_churn
from inference.decision_engine import retention_decision

app = FastAPI(title="Customer Churn Prediction API")

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(customer: dict):
    churn_prob = predict_churn(customer)
    decision = retention_decision(churn_prob)

    return {
        "churn_probability": round(churn_prob, 3),
        "recommended_action": decision
    }
