# ========================================
# app.py — Streamlit Web App for Heart Disease Prediction
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

import os
import urllib.request

# URL to your xgb_model.pkl on GitHub (raw link, not webpage!)
model_url = "https://raw.githubusercontent.com/lmd0420/bis568/main/project/xgb_model.pkl"
model_path = "xgb_model.pkl"

# Download if not exists
if not os.path.exists(model_path):
    urllib.request.urlretrieve(model_url, model_path)

# Now load the model
import joblib
model = joblib.load(model_path)

# ========================================
# Streamlit UI
# ========================================
st.title("❤️ Heart Disease Prediction App")

st.write("Fill out the form below to predict if you have heart disease risk.")

# Define input fields
age = st.slider('Age', 20, 100, 50)
sex = st.selectbox('Sex', options=['Male', 'Female'])
chest_pain_type = st.selectbox('Chest Pain Type', options=['TA', 'ATA', 'NAP', 'ASY'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=200, value=120)
cholesterol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
resting_ecg = st.selectbox('Resting ECG', options=['Normal', 'ST', 'LVH'])
max_hr = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
exercise_angina = st.selectbox('Exercise Induced Angina', options=['Yes', 'No'])
oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st_slope = st.selectbox('ST Slope', options=['Up', 'Flat', 'Down'])

# ========================================
# Map Inputs to Model Format
# ========================================
if st.button('Predict'):

    # Label encoding (must match your training!)
    sex_mapping = {'Male': 1, 'Female': 0}
    chest_pain_mapping = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
    resting_ecg_mapping = {'Normal': 0, 'ST': 1, 'LVH': 2}
    exercise_angina_mapping = {'No': 0, 'Yes': 1}
    st_slope_mapping = {'Up': 0, 'Flat': 1, 'Down': 2}

    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_mapping[sex]],
        'ChestPainType': [chest_pain_mapping[chest_pain_type]],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg_mapping[resting_ecg]],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina_mapping[exercise_angina]],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_mapping[st_slope]],
    })

    # Predict
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    # Output
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ High risk of Heart Disease! (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Low risk of Heart Disease! (Confidence: {1-probability:.2f})")
