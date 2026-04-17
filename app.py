import streamlit as st
import joblib
import pandas as pd

# Load model and label
model = joblib.load('C:/Users/pooja/OneDrive/Desktop/ML Internship/Day 10/Finial Project/Weight analysis/weight.pkl')
label = joblib.load('C:/Users/pooja/OneDrive/Desktop/ML Internship/Day 10/Finial Project/Weight analysis/label.pkl')

# Page settings
st.set_page_config(page_title="Health Analyzer", page_icon="💪", layout="centered")

st.title("❤️‍🩹 Health Analyzer")
st.write("Enter your details below to analyze your health status based on height, weight, and age.")

# User Inputs
age = st.number_input("Enter your Age", min_value=1, max_value=120, step=1)
height = st.number_input("Enter your Height (in cm)", min_value=50, max_value=250, step=1)
weight = st.number_input("Enter your Weight (in kg)", min_value=10, max_value=300, step=1)

# Predict Button
if st.button("Analyze My Health"):
    input_df = pd.DataFrame([[age, height, weight]], columns=['age', 'height', 'weight'])
    prediction = model.predict(input_df)
    status = label.inverse_transform(prediction)[0]

    height_m = height / 100
    bmi = weight / (height_m ** 2)
    min_weight = 18.5 * (height_m ** 2)
    max_weight = 24.9 * (height_m ** 2)

    st.subheader("📊 Result")
    st.markdown(f"**Health Status:** `{status}`")
    #st.markdown(f"**BMI:** `{bmi:.2f}`")

    if weight < min_weight:
        st.warning(f"⚠️ You are underweight. Ideal weight: **{min_weight:.1f} – {max_weight:.1f} kg**")
    elif weight > max_weight:
        st.warning(f"⚠️ You are overweight. Ideal weight: **{min_weight:.1f} – {max_weight:.1f} kg**")
    else:
        st.success("✅ You are in a healthy weight range!")

    if height > (age * 5 + 80):
        st.info("ℹ️ Your height is quite high compared to your age.")

st.markdown("---")
st.caption("Created by PoojaRajaram💫 using Machine Learning ")
