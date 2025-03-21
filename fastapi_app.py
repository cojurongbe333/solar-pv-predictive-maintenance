# 🚀 FastAPI ML Model Deployment for Solar PV Predictive Maintenance

from fastapi import FastAPI
import pandas as pd
import joblib
import uvicorn
import numpy as np
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load trained ML model
model = joblib.load("../models/solar_pv_failure_model.pkl")  # Replace with actual model path

# Define request body schema
class SensorData(BaseModel):
    voltage: float
    temperature: float
    irradiance: float
    humidity: float
    wind_speed: float

# Define prediction endpoint
@app.post("/predict")
def predict_failure(data: SensorData):
    try:
        # Convert input data to model format
        input_features = np.array([[data.voltage, data.temperature, data.irradiance, data.humidity, data.wind_speed]])
        prediction = model.predict(input_features)

        # Format response
        return {
            "prediction": "Failure Detected" if prediction[0] == 1 else "No Failure",
            "confidence": float(prediction[0])
        }
    
    except Exception as e:
        return {"error": str(e)}

# Run API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
