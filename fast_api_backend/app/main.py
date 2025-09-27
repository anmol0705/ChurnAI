# app/main.py
from fastapi import FastAPI
from .schemas import ChurnFeatures, PredictionOut
from .model import churn_model

app = FastAPI(
    title="Customer Churn Prediction API",
    description="An API to predict customer churn based on their banking details.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionOut)
def predict_churn(features: ChurnFeatures):
    """
    Takes customer details and predicts churn.
    """
    prediction_result = churn_model.predict(features)
    return prediction_result