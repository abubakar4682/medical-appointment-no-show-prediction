import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Medical Appointment No-Show Prediction",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# S3 model URLs
# -----------------------------
MODEL_URL = "https://appointment-noshow-project.s3.eu-west-2.amazonaws.com/models/rf_model.pkl"
COLUMNS_URL = "https://appointment-noshow-project.s3.eu-west-2.amazonaws.com/models/feature_columns.pkl"

# -----------------------------
# Load model and feature columns from S3
# -----------------------------
@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

@st.cache_resource
def load_feature_columns():
    response = requests.get(COLUMNS_URL)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

model = load_model()
feature_columns = load_feature_columns()

# -----------------------------
# App title and description
# -----------------------------
st.title("🏥 Medical Appointment No-Show Prediction")

st.write(
    """
    This app predicts whether a patient is likely to miss a medical appointment.
    The model was trained using the Medical Appointment No-Show dataset and deployed
    using AWS S3 and Streamlit.
    """
)

st.warning(
    "Educational project only. This app should not be used as a real medical decision-making system."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Project Summary")
st.sidebar.write("**Dataset:** Medical Appointment No-Show")
st.sidebar.write("**Rows:** 110,527")
st.sidebar.write("**Best model:** Random Forest")
st.sidebar.write("**Model storage:** AWS S3")
st.sidebar.write("**Key feature:** Wait days")

# -----------------------------
# User inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Details")
    age = st.slider("Age", 0, 100, 30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    sms_received = st.selectbox("SMS Received", [0, 1])
    wait_days = st.slider("Wait Days", 0, 100, 5)

with col2:
    st.subheader("Health and Social Factors")
    scholarship = st.selectbox("Scholarship", [0, 1])
    hypertension = st.selectbox("Hypertension", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])
    alcoholism = st.selectbox("Alcoholism", [0, 1])
    handicap = st.selectbox("Handicap", [0, 1])

# -----------------------------
# Prepare input data
# -----------------------------
input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

input_data["age"] = age
input_data["sms_received"] = sms_received
input_data["wait_days"] = wait_days
input_data["scholarship"] = scholarship
input_data["hypertension"] = hypertension
input_data["diabetes"] = diabetes
input_data["alcoholism"] = alcoholism
input_data["handicap"] = handicap

# Gender encoding
if gender == "Male" and "gender_M" in input_data.columns:
    input_data["gender_M"] = 1

# Age group encoding
if age <= 12:
    if "age_group_child" in input_data.columns:
        input_data["age_group_child"] = 1
elif age <= 18:
    if "age_group_teen" in input_data.columns:
        input_data["age_group_teen"] = 1
elif age <= 35:
    if "age_group_young_adult" in input_data.columns:
        input_data["age_group_young_adult"] = 1
elif age <= 60:
    if "age_group_adult" in input_data.columns:
        input_data["age_group_adult"] = 1
else:
    if "age_group_senior" in input_data.columns:
        input_data["age_group_senior"] = 1

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("Predict Appointment Outcome"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric("No-show Probability", f"{probability:.2%}")

    with result_col2:
        if prediction == 1:
            st.metric("Prediction", "Likely No-Show")
        else:
            st.metric("Prediction", "Likely Attend")

    st.progress(float(probability))

    if prediction == 1:
        st.error("Patient is likely to MISS the appointment.")
    else:
        st.success("Patient is likely to ATTEND the appointment.")

    st.subheader("Input Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Age",
            "Gender",
            "SMS Received",
            "Wait Days",
            "Scholarship",
            "Hypertension",
            "Diabetes",
            "Alcoholism",
            "Handicap"
        ],
        "Value": [
            age,
            gender,
            sms_received,
            wait_days,
            scholarship,
            hypertension,
            diabetes,
            alcoholism,
            handicap
        ]
    })

    st.dataframe(summary, use_container_width=True)

# -----------------------------
# Model information
# -----------------------------
st.divider()

st.subheader("Model Performance Summary")

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [0.79, 0.77, 0.80],
    "Recall (No-show)": [0.02, 0.22, 0.05],
    "F1-score (No-show)": [0.03, 0.29, 0.09]
})

st.dataframe(results, use_container_width=True)

st.info(
    """
    Random Forest was selected because it performed better at detecting actual
    no-show patients compared with Logistic Regression and XGBoost.
    """
)

st.subheader("Key Insight")

st.write(
    """
    Feature importance analysis showed that **wait_days** was the strongest predictor.
    This means patients with longer waiting times were more likely to miss appointments.
    """
)