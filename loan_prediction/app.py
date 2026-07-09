import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==============================
# Load Model, Scaler, Encoders, and Schema
# ==============================


BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "loan_prediction_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
encoders = joblib.load(BASE_DIR / "encoders.pkl")
feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
categorical_columns = joblib.load(BASE_DIR / "categorical_columns.pkl")

# ==============================
# Title
# ==============================

st.title("🏦 Loan Default Prediction System")
st.write("Enter Customer Details")

# ==============================
# Widgets for known numeric fields (nicer labels/ranges than a generic box)
# ==============================

numeric_widgets = {
    "customer_age": lambda: st.number_input("Customer Age", 18, 100, 30),
    "customer_income": lambda: st.number_input("Customer Income", 1000, 1000000, 50000),
    "employment_duration": lambda: st.number_input("Employment Duration (years)", 0, 40, 5),
    "loan_amnt": lambda: st.number_input("Loan Amount", 1000, 1000000, 100000),
    "loan_int_rate": lambda: st.number_input("Interest Rate (%)", 1.0, 30.0, 10.0),
    "cred_hist_length": lambda: st.number_input("Credit History Length (years)", 1, 50, 5),
    "term_years": lambda: st.selectbox("Loan Term (years)", [3, 5, 7]),
}

# ==============================
# Build inputs by walking the SAME column list/order used in training
# ==============================

inputs = {}

for col in feature_columns:
    if col in categorical_columns:
        # This column was LabelEncoded during training -> show the real
        # category names and encode the choice with the SAME encoder,
        # so the numeric code matches what the model was trained on.
        label = col.replace("_", " ").title()
        choice = st.selectbox(label, list(encoders[col].classes_))
        inputs[col] = encoders[col].transform([choice])[0]
    elif col in numeric_widgets:
        inputs[col] = numeric_widgets[col]()
    else:
        # Fallback for any column not explicitly covered above
        inputs[col] = st.number_input(col.replace("_", " ").title(), value=0.0)

# ==============================
# Prediction
# ==============================

if st.button("Predict"):

    # Build the row and force it into the EXACT column order the scaler
    # and model were fit on -- this is what fixes the mismatch.
    data = pd.DataFrame([inputs])[feature_columns]

    # Scale Input Data
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)

    # Display Result
    if prediction[0] == 1:
        st.error("⚠ Loan Will Default")
    else:
        st.success("✅ Loan Will NOT Default")
