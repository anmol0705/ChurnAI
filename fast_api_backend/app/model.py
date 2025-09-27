# app/model.py
import pickle
import pandas as pd
from .schemas import ChurnFeatures

class ChurnModel:
    def __init__(self, model_path: str):
        """Initializes by loading the model and scaler from the .pkl file."""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']

    def predict(self, features: ChurnFeatures) -> dict:
        """
        Performs the full preprocessing pipeline from the notebook
        and returns a prediction.
        """
        # 1. Create a DataFrame from the input
        input_data = features.dict()
        df = pd.DataFrame([input_data])

        # 2. Replicate the encoding from your notebook
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        geography_dummies = pd.get_dummies(df['Geography'], prefix='Geography').reindex(
            columns=['Geography_France', 'Geography_Germany', 'Geography_Spain'], fill_value=0
        )
        df = pd.concat([df, geography_dummies], axis=1)

        # 3. Replicate the feature engineering
        df['Balance_Salary_Ratio'] = (df['Balance'] / df['EstimatedSalary']).fillna(0)

        # 4. Drop the original columns -- THIS IS THE CORRECTED LINE
        df = df.drop(columns=['Geography', 'Balance', 'EstimatedSalary'])
        
        # 5. Ensure column order is exactly as it was during training
        final_feature_order = [
            'CreditScore', 'Gender', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard',
            'IsActiveMember', 'Geography_France', 'Geography_Germany', 'Geography_Spain',
            'Balance_Salary_Ratio'
        ]
        df = df[final_feature_order]

        # 6. Apply the scaler
        scaled_features = self.scaler.transform(df)
        
        # 7. Make the final prediction
        prediction = self.model.predict(scaled_features)[0]

        return {"churn_prediction": int(prediction)}

# Instantiate the model with the path to your file
churn_model = ChurnModel(model_path="D:\\AI-ML course\\Churn Model\\fast_api_backend\\models\\churn_model.pkl")