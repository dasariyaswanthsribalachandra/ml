from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load model
model = joblib.load("model.pkl")

# Request schema
class InputData(BaseModel):
    number: float

# Health check
@app.get("/")
def home():
    return {"message": "ML Model API Running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):

    input_value = np.array([[data.number]])

    prediction = model.predict(input_value)

    return {
        "input": data.number,
        "prediction": prediction[0]
    }