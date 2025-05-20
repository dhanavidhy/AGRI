import streamlit as st
import joblib
import numpy as np

# Load the trained model and encoders
model = joblib.load('stacked_model.pkl')
scaler = joblib.load('scaler.pkl')
le_state = joblib.load('le_state.pkl')
le_district = joblib.load('le_district.pkl')
le_season = joblib.load('le_season.pkl')
le_crop = joblib.load('le_crop.pkl')

# App title
st.title("🌾 Crop Yield Prediction App")
st.markdown("Enter the details below to predict the expected crop production (in tonnes).")

# User input fields (with selectbox to avoid unseen label errors)
state = st.selectbox("State", le_state.classes_)
district = st.selectbox("District", le_district.classes_)
season = st.selectbox("Season", le_season.classes_)
crop = st.selectbox("Crop", le_crop.classes_)
area = st.number_input("Area of Cultivation (in hectares)", min_value=0.1)

# Prediction button
if st.button("Predict Yield"):
    try:
        # Encode user input
        state_enc = le_state.transform([state])[0]
        district_enc = le_district.transform([district])[0]
        season_enc = le_season.transform([season])[0]
        crop_enc = le_crop.transform([crop])[0]

        # Create input array
        input_data = np.array([[state_enc, district_enc, season_enc, crop_enc, area]])
        input_scaled = scaler.transform(input_data)

        # Get prediction
        with st.spinner("Predicting..."):
            prediction = model.predict(input_scaled)[0]

        st.success(f"🌾 Predicted Crop Production: **{prediction:.2f} tonnes**")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Please verify your inputs. If the problem persists, contact the developer.")
