from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from fastapi.responses import HTMLResponse

app = FastAPI()

# Load the pre-trained model (ensure it's saved as a .pkl file)
model = joblib.load("rf_model_tuned5.pkl")

# Define normal values for each variable
normal_values = {
    "MemoryComplaints": 0,
    "Forgetfulness": 0,
    "Disorientation": 0,
    "Confusion": 0,
    "Depression": 0,
    "FamilyHistoryAlzheimers": 0,
    "Age": 65,
    "BehavioralProblems": 0,
    "CardiovascularDisease": 0,
    "PhysicalActivity": 8
}

class UserInput(BaseModel):
    MemoryComplaints: conint(ge=0, le=1)
    Forgetfulness: conint(ge=0, le=1)
    Disorientation: conint(ge=0, le=1)
    Confusion: conint(ge=0, le=1)
    Depression: conint(ge=0, le=1)
    FamilyHistoryAlzheimers: conint(ge=0, le=1)
    Age: int
    BehavioralProblems: conint(ge=0, le=1)
    CardiovascularDisease: conint(ge=0, le=1)
    PhysicalActivity: int

@app.post("/predict")
def predict(input: UserInput):
    # Prepare input data for prediction
    input_data = pd.DataFrame([input.dict()])
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    risk = "High Risk of Alzheimer's" if prediction[0] == 1 else "Low Risk of Alzheimer's"
    
    return {"predicted_risk": risk}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Alzheimer's Detection Tool!"}

@app.get("/comparison")
def comparison():
    # Generate comparison plots (bar charts) based on normal values and user inputs
    fig, ax = plt.subplots()
    
    categories = list(normal_values.keys())
    user_values = [normal_values[cat] for cat in categories]
    
    ax.bar(categories, user_values, color=['skyblue', 'green'])
    
    ax.set_title("Comparison with Normal Values")
    ax.set_ylabel("Value")
    
    # Save the plot to a file
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig("comparison_plot.png")
    
    return HTMLResponse(content="<img src='/static/comparison_plot.png'>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
