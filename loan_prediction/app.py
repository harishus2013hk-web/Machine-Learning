import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load Model and Scaler
# ==============================

model = joblib.load("loan_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# Title
# ==============================

st.title("🏦 Loan Default Prediction System")

st.write("Enter Customer Details")

# ==============================
# User Input
# ==============================

customer_age = st.number_input("Customer Age", 18, 100, 30)

customer_income = st.number_input("Customer Income", 1000, 1000000, 50000)

employment_duration = st.number_input("Employment Duration", 0, 40, 5)

loan_amnt = st.number_input("Loan Amount", 1000, 1000000, 100000)

loan_int_rate = st.number_input("Interest Rate (%)", 1.0, 30.0, 10.0)

cred_hist_length = st.number_input("Credit History Length", 1, 50, 5)

# ==============================
# Prediction
# ==============================

if st.button("Predict"):

    # Create DataFrame
    data = pd.DataFrame({
        "customer_age": [customer_age],
        "customer_income": [customer_income],
        "employment_duration": [employment_duration],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "cred_hist_length": [cred_hist_length]
    })

    # Scale Input Data
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)

    # Display Result
    if prediction[0] == 1:
        st.error("⚠ Loan Will Default")
    else:
        st.success("✅ Loan Will NOT Default")