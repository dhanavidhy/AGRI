import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load('stacked_model.pkl')
scaler = joblib.load('scaler.pkl')
le_state = joblib.load('le_state.pkl')
le_district = joblib.load('le_district.pkl')
le_season = joblib.load('le_season.pkl')
le_crop = joblib.load('le_crop.pkl')

# Load dataset to get dropdown options
df = pd.read_csv("merged_crop_data.csv")

# Get unique sorted values for dropdowns
states = sorted(df['State_Name'].unique())
districts = sorted(df['District_Name'].unique())
crops = sorted(df['Crop'].unique())
seasons = ['Kharif', 'Rabi', 'Whole Year', 'Autumn', 'Summer', 'Winter']  # Preset seasons

# App title
st.title("🌾 Crop Yield Prediction App")

st.markdown("Enter the details below to predict the expected crop production (in tonnes).")

# User inputs with dropdowns
state = st.selectbox("Select State", states)
district = st.selectbox("Select District", districts)
season = st.selectbox("Select Season", seasons)
crop = st.selectbox("Select Crop", crops)
area = st.number_input("Area of Cultivation (in hectares)", min_value=0.1)

# Prediction
if st.button("Predict Yield"):
    try:
        # Encode inputs
        state_enc = le_state.transform([state])[0]
        district_enc = le_district.transform([district])[0]
        season_enc = le_season.transform([season])[0]
        crop_enc = le_crop.transform([crop])[0]

        # Prepare input for prediction
        input_data = np.array([[state_enc, district_enc, season_enc, crop_enc, area]])
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        st.success(f"🌾 Predicted Crop Production: **{prediction:.2f} tonnes**")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Ensure all selected values exist in the trained dataset.")