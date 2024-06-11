import streamlit as st
import joblib
import numpy as np

# Load the models
random_forest_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.markdown("<h1 style='text-align: center; color: #00796b;'>Predict Environment</h1>", unsafe_allow_html=True)

# Create the form
HS_analog = st.text_input('HS Analog:')
L_lux = st.text_input('L Lux:')
T_deg = st.text_input('Temperature (°C):')
CO2_analog = st.text_input('CO2 Analog:')
HR_percent = st.text_input('Humidity (%):')

if st.button('Predict'):
    try:
        # Prepare the feature array
        features = np.array([[HS_analog, L_lux, T_deg, CO2_analog, HR_percent]]).astype(float)

        # Scale the features
        scaled_features = scaler.transform(features)

        # Make the prediction
        prediction = random_forest_model.predict(scaled_features)

        # Interpret the prediction
        if prediction == 1:
            result = "Dry soil"
        elif prediction == 2:
            result = "Good environment"
        elif prediction == 3:
            result = "Too hot"
        elif prediction == 4:
            result = "Too cold environment"
        else:
            result = "Unknown"
        
        st.success(f'Result: {result}')
    except Exception as e:
        st.error(f"Error: {e}")

# Styling with Markdown
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e0f7fa;
    }
    h1 {
        text-align: center;
        color: #00796b;
    }
    label {
        margin-top: 15px;
        color: #004d40;
        font-size: 1.1em;
    }
    .stTextInput > div > div > input {
        padding: 10px;
        margin-top: 5px;
        border: 1px solid #b2dfdb;
        border-radius: 5px;
        font-size: 1em;
    }
    button {
        margin-top: 20px;
        padding: 10px;
        background-color: #00796b;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1.1em;
        transition: background-color 0.3s ease;
    }
    button:hover {
        background-color: #004d40;
    }
    .result {
        margin-top: 20px;
        padding: 10px;
        text-align: center;
        background-color: #e0f2f1;
        border: 1px solid #b2dfdb;
        border-radius: 5px;
        font-size: 1.2em;
    }
    </style>
    """,
    unsafe_allow_html=True
)
