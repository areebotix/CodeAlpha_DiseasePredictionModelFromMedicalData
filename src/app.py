import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Heart Health Risk Checker", layout="centered")

st.title("❤️ Heart Health Risk Checker")
st.write("Enter your health details to estimate heart disease risk.")
st.caption("⚠️ This is an AI-based educational tool, not a medical diagnosis.")

# ---------------------------
# LOAD MODEL
# ---------------------------
#model = joblib.load("C:/CodeAlpha_DiseasePredictionFromMedicalData/models/heart_model.pkl")
model = joblib.load("models/heart_model.pkl")
# ---------------------------
# USER-FRIENDLY INPUTS (NO RAW CODES)
# ---------------------------

age = st.slider("Your Age", 20, 100, 40)

gender = st.selectbox("Gender", ["Female", "Male"])
sex = 1 if gender == "Male" else 0

chest_pain = st.selectbox(
    "Chest Pain Type",
    [
        "No pain",
        "Mild discomfort (atypical)",
        "Chest pain during normal activity",
        "Severe/Resting chest pain"
    ]
)
cp_map = {
    "No pain": 0,
    "Mild discomfort (atypical)": 1,
    "Chest pain during normal activity": 2,
    "Severe/Resting chest pain": 3
}
cp = cp_map[chest_pain]

blood_pressure = st.slider("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.slider("Cholesterol Level", 100, 600, 200)

fasting_sugar = st.selectbox("High Blood Sugar (>120 mg/dl)?", ["No", "Yes"])
fbs = 1 if fasting_sugar == "Yes" else 0

ecg_result = st.selectbox(
    "Resting ECG Result",
    ["Normal", "Minor abnormality", "Major abnormality"]
)
restecg = {"Normal": 0, "Minor abnormality": 1, "Major abnormality": 2}[ecg_result]

max_heart_rate = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

exercise_angina = st.selectbox("Pain during exercise?", ["No", "Yes"])
exang = 1 if exercise_angina == "Yes" else 0

oldpeak = st.slider("Stress Test Indicator (Oldpeak)", 0.0, 6.0, 1.0)

slope_type = st.selectbox(
    "Heart Stress Response",
    ["Normal", "Flat", "High risk slope"]
)
slope = {"Normal": 0, "Flat": 1, "High risk slope": 2}[slope_type]

vessels = st.selectbox(
    "Blocked Arteries Level",
    ["None", "Low", "Medium", "High"]
)
ca = {"None": 0, "Low": 1, "Medium": 2, "High": 3}[vessels]

thal_result = st.selectbox(
    "Blood Flow Test Result",
    ["Normal", "Fixed defect", "Reversible defect"]
)
thal = {"Normal": 0, "Fixed defect": 1, "Reversible defect": 2}[thal_result]

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("Check Heart Risk"):

    input_data = np.array([[age, sex, cp, blood_pressure, cholesterol, fbs,
                            restecg, max_heart_rate, exang, oldpeak,
                            slope, ca, thal]])

    probability = model.predict_proba(input_data)[0][1]

    # Risk logic
    if probability < 0.3:
        risk = "🟢 Low Risk"
    elif probability < 0.7:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    st.subheader("Result")

    st.metric("Heart Disease Probability", f"{round(probability*100,2)}%")
    st.success(f"Risk Level: {risk}")

    # ---------------------------
    # SHAP EXPLANATION
    # ---------------------------
    
import shap
import matplotlib.pyplot as plt
import streamlit as st

st.subheader("Why this prediction? (AI explanation)")

try:
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots()

    shap.summary_plot(
        shap_values,
        input_data,
        show=False
    )

    st.pyplot(plt.gcf())

except Exception as e:
    st.warning("Explainability visualization not available.")
    st.text(str(e))
