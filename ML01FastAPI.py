from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model = joblib.load('purchase_predictor_model.pkl')

# Define an enumeration for gender
class Gender(str, Enum):
    male = "Male"
    female = "Female"

# Define the input data model using Enum for gender
class UserInput(BaseModel):
    age: int
    gender: Gender  # Use the Gender enum

@app.get("/")
def read_root():
    return {"message": "Welcome to the E-commerce Purchase Prediction API!"}

@app.post("/predict")
def predict(input: UserInput):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Age': [input.age],
        'Gender_Male': [1 if input.gender == 'Male' else 0],
        # Add other preprocessed features here based on user input
    })

    try:
        # Predict purchase amount
        prediction = model.predict(input_data)
        return {"predicted_purchase_amount": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "Healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
