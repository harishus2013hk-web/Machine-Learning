import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Hospital Patient Status Prediction",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# Load Model and Encoders
# ==========================================
model = joblib.load("hospital_status_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# ==========================================
# Sidebar
# ==========================================
if os.path.exists("hospital.png"):
    st.sidebar.image("hospital.png", use_container_width=True)

st.sidebar.title("🏥 Hospital Dashboard")
st.sidebar.write("Patient Status Prediction")

st.sidebar.metric("Model", "Random Forest")
st.sidebar.metric("Accuracy", "98.5%")

st.sidebar.markdown("---")
st.sidebar.write("Developed by")
st.sidebar.write("**Harish Kumar**")

# ==========================================
# Main Title
# ==========================================
st.title("🏥 Hospital Patient Status Prediction")
st.write("Fill in the patient details below and click **Predict**.")

st.markdown("---")

# ==========================================
# Input Form
# ==========================================

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 35)

    los = st.number_input(
        "Length of Stay (LOS)",
        1,
        60,
        5
    )

    er_time = st.number_input(
        "ER Time",
        0,
        500,
        60
    )

    cost = st.number_input(
        "Treatment Cost",
        0.0,
        1000000.0,
        1000.0
    )

with col2:

    gender = st.selectbox(
        "Gender",
        encoders["Gender"].classes_
    )

    patient_type = st.selectbox(
        "Patient Type",
        encoders["Patient type"].classes_
    )

    department = st.selectbox(
        "Department",
        encoders["Department_Name"].classes_
    )

    disease = st.selectbox(
        "Disease",
        encoders["disease_name"].classes_
    )

st.markdown("---")

# ==========================================
# Prediction
# ==========================================

if st.button("🔍 Predict Patient Status"):

    gender_encoded = encoders["Gender"].transform([gender])[0]

    patient_type_encoded = encoders["Patient type"].transform([patient_type])[0]

    department_encoded = encoders["Department_Name"].transform([department])[0]

    disease_encoded = encoders["disease_name"].transform([disease])[0]

    input_data = pd.DataFrame({
        "Age":[age],
        "LOS":[los],
        "ER_Time":[er_time],
        "treatmentcost":[cost],
        "Gender":[gender_encoded],
        "Patient type":[patient_type_encoded],
        "Department_Name":[department_encoded],
        "disease_name":[disease_encoded]
    })

    prediction = model.predict(input_data)

    status = encoders["Status"].inverse_transform(prediction)[0]

    st.success(f"### ✅ Predicted Patient Status : {status}")

    # ======================================
    # Prediction Probability
    # ======================================

    if hasattr(model, "predict_proba"):

        prob = model.predict_proba(input_data)[0]

        probability = pd.DataFrame({
            "Status": encoders["Status"].classes_,
            "Probability (%)": (prob * 100).round(2)
        })

        st.subheader("Prediction Probability")

        st.dataframe(
            probability,
            use_container_width=True
        )

    # ======================================
    # Patient Summary
    # ======================================

    st.subheader("Patient Details")

    summary = pd.DataFrame({
        "Feature":[
            "Age",
            "LOS",
            "ER Time",
            "Treatment Cost",
            "Gender",
            "Patient Type",
            "Department",
            "Disease"
        ],
        "Value":[
            age,
            los,
            er_time,
            cost,
            gender,
            patient_type,
            department,
            disease
        ]
    })

    st.table(summary)

st.markdown("---")
st.caption("Hospital Patient Status Prediction System | Streamlit + Scikit-learn")