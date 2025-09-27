# app/schemas.py
from pydantic import BaseModel, Field
from typing import Literal

class ChurnFeatures(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int = Field(..., ge=0, le=1) # Must be 0 or 1
    IsActiveMember: int = Field(..., ge=0, le=1) # Must be 0 or 1
    EstimatedSalary: float
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]

    class Config:
        schema_extra = {
            "example": {
                "CreditScore": 650, "Age": 42, "Tenure": 2, "Balance": 125000.0,
                "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1,
                "EstimatedSalary": 100000.0, "Geography": "France", "Gender": "Female"
            }
        }

class PredictionOut(BaseModel):
    churn_prediction: int