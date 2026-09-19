import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("diabetes_knn.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

st.title("Diabetes Prediction")

age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=120
)

glucose = st.number_input(
    "Enter Glucose Level",
    min_value=0.0
)

bmi = st.number_input(
    "Enter BMI",
    min_value=0.0
)

if st.button("Predict"):

    input_data = np.array([[age, glucose, bmi]])

    # Scale input using the same scaler
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Diabetes")
    else:
        st.success("No Diabetes")