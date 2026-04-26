from pathlib import Path
import streamlit as st
import pickle
import numpy as np

BASE_DIR = Path(__file__).parent
# Load model
with open(BASE_DIR / 'model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open(BASE_DIR / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Advertising Sales Prediction")
st.write("Enter the advertising spend for TV, radio and newspaper to predict sales.")

tv = st.number_input("TV Advertising Spend in $", min_value=0.0)
radio = st.number_input("Radio Advertising Spend in $", min_value=0.0)
newspaper = st.number_input("Newspaper Advertising Spend in $", min_value=0.0)

if st.button("Predict Sales"):
    input_data = np.array([[tv, radio, newspaper]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(f"Predicted Sales: {prediction[0]:.2f}")
