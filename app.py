import streamlit as st
import pickle
import numpy as np
with open('model.pkl','rb') as f:
    model=pickle.load(f)
st.title("Advertising Sales Prediction")
st.write("Enter the advertising spend for TV, radio and newspaper to predict sales.")

tv=st.number_input("TV Advertising Spend in $",min_value=0.0)
radio=st.number_input("Radio Advertising Spend in $",min_value=0.0)
newspaper=st.number_input("Newspaper Advertising Spend in $",min_value=0.0)

if st.button("Predict Sales"):
    input_data=np.array([[tv,radio,newspaper]])
    prediction=model.predict(input_data)
    st.success(f"Predicted Sales: {prediction[0]:.2f}")