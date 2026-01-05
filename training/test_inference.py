from inference.predict import predict_churn
from inference.decision_engine import retention_decision

# Sample customer input
sample_customer = {
    "gender": 0,               # REQUIRED (0 = Female, 1 = Male after encoding)
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 2,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 1,
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 0,
    "Contract": 0,
    "PaperlessBilling": 1,
    "PaymentMethod": 2,
    "MonthlyCharges": 85.5,
    "TotalCharges": 170.2
}


prob = predict_churn(sample_customer)
decision = retention_decision(prob)

print("Churn Probability:", round(prob, 3))
print("Business Decision:", decision)
