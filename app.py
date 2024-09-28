import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load the model using joblib
model = joblib.load('model.joblib')  # Ensure this file is in the same directory

# Function to predict the ride cost
def predict_price(number_of_riders, number_of_drivers, vehicle_type, expected_ride_duration):
    vehicle_type_mapping = {"Premium": 1, "Economy": 0}
    vehicle_type_numeric = vehicle_type_mapping.get(vehicle_type)

    if vehicle_type_numeric is None:
        raise ValueError("Invalid vehicle type")

    input_data = np.array([[number_of_riders, number_of_drivers, vehicle_type_numeric, expected_ride_duration]])
    predicted_price = model.predict(input_data)
    return predicted_price

# Streamlit UI
st.title("Ride Cost Prediction Using Dynamic Pricing")

# Input fields for user data
number_of_riders = st.number_input("Enter the number of riders", min_value=0)
number_of_drivers = st.number_input("Enter the number of drivers", min_value=0)
vehicle_type = st.selectbox("Select vehicle type", ("Premium", "Economy"))
expected_ride_duration = st.number_input("Enter the expected ride duration in minutes", min_value=0)

# Predict button
if st.button("Predict Ride Cost"):
    # Make the prediction
    try:
        predicted_cost = predict_price(number_of_riders, number_of_drivers, vehicle_type, expected_ride_duration)
        st.success(f"Predicted Ride Cost: ${predicted_cost[0]:.2f}")
    except ValueError as e:
        st.error(e)
