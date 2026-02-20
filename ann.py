import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load your scaler and model (save them from your notebook)
scaler = joblib.load('scaler.pkl')
model = load_model('churn_ann_model.h5')

st.title("Customer Churn Prediction App")

st.header("Enter Customer Information")
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=92, value=35)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=0.0)
num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Encoding helpers
geo_map = {"France": 0, "Germany": 1, "Spain": 2}
gender_map = {"Male": 0, "Female": 1}

if st.button('Predict'):
    features = np.array([[credit_score,
                          geo_map[geography],
                          gender_map[gender],
                          age,
                          tenure,
                          balance,
                          num_products,
                          has_cr_card,
                          is_active_member,
                          estimated_salary]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    outcome = "Churn" if prediction[0][0] > 0.5 else "Not Churn"
    st.subheader(f"Prediction: {outcome}")
