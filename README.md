# Customer Churn Prediction – End-to-End MLOps

## Problem Statement
Customer churn is a major business problem for telecom companies.
This project predicts whether a customer will churn using machine learning
and exposes the prediction through a FastAPI service.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Docker

## Project Workflow
1. Data ingestion and preprocessing
2. Feature engineering
3. Model training
4. Model evaluation
5. Model persistence
6. Prediction using FastAPI
7. Containerization using Docker

## Folder Structure
- `training/` : model training and evaluation scripts
- `inference/` : prediction logic
- `api/` : FastAPI application
- `data/` : dataset files
- `models/` : saved trained models

## Model Performance
- Classification problem: churn vs no churn
- Metric used: Accuracy / ROC-AUC  
- Reason: suitable for binary classification problems

## How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python training/train.py
```

### 3. Run the API
```bash
uvicorn api.main:app --reload
```

### 4. Run with Docker
```bash
docker build -t churn-mlops .
docker run -p 8000:8000 churn-mlops
```

## Key Learnings
- Built an end-to-end machine learning pipeline
- Deployed ML model using FastAPI
- Understood basic MLOps practices including containerization

