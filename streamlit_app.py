# Streamlit App for Solar PV Predictive Maintenance

import streamlit as st
import numpy as np
import joblib

# Load model (replace with actual model path)
model = joblib.load("models/solar_pv_failure_model.pkl")

st.title("Solar PV Predictive Maintenance")
st.write("Enter sensor values below to predict if a failure will occur.")

# Input fields
voltage = st.number_input("Voltage (V)", min_value=0.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=-50.0, step=0.1)
irradiance = st.number_input("Irradiance (W/m²)", min_value=0.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, step=0.1)
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, step=0.1)

if st.button("Predict"):
    input_data = np.array([[voltage, temperature, irradiance, humidity, wind_speed]])
    prediction = model.predict(input_data)[0]
    result = "Failure Detected" if prediction == 1 else "No Failure"
    st.success(f"Prediction: {result}")
