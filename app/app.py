# app/app.py
import streamlit as st
import joblib
import numpy as np

# --- Load model and scaler ---
model = joblib.load("app/xgb_model.joblib")
scaler = joblib.load("app/scaler.joblib")  # scaler used during training

# --- Define transformed features ---
log1p_features = ["physical_activity_time", "poverty_ratio"]
log_features = ["bmi", "waist_perimeter", "weight", "cholesterol"]
sqrt_sym_features = ["weight_diff"]

# --- Mean values for features not provided by user ---
X_mean = {
    "poverty_ratio": 2.3,
    "family_income": 4.6,
    "health_insurance": 0.75,
    "physical_activity_time": 7.37,
    "systolic_bp": 113.1,
    "diastolic_bp": 64.85,
    "pulse_rate": 72.96,
    "waist_perimeter": 90.0,
    "weight": 75.0,
    "height": 167.2,
    "cholesterol": 190.0,
    "red_blood_cell_count": 4.81,
    "hemoglobin": 14.32,
    "hematocrit": 42.3,
    "mean_cell_hemoglobin": 29.82,
    "red_cell_distribution_width": 12.51,
    "weight_diff": 0.0,
    # one-hot categorical averages
    "ethnic_mexican": 0.316,
    "ethnic_hispanic": 0.044,
    "ethnic_white": 0.338,
    "ethnic_black": 0.264,
    "ethnic_other": 0.038,
    "smoker_former": 0.24,
    "smoker_no": 0.55,
    "smoker_yes": 0.21,
}

# --- Streamlit UI ---
st.title("VO₂max Prediction App 🫁")
st.write(
    "Enter your biometric information to estimate your VO₂max. "
    "The model was trained on participants aged 16–49 years. Predictions outside this range may be less reliable."
)

# --- User inputs ---
age = st.number_input("Age (years)", min_value=10, max_value=100, value=25)
sex = st.selectbox("Sex", options=["Male", "Female"])
body_fat_percent = st.number_input("Body Fat Percentage (%)", min_value=2.0, max_value=61.8, value=30.0)
bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=50.0, value=22.0)
education_level = st.selectbox(
    "Education Level",
    options=["Less than high school", "High school", "Greater than high school"]
)
smoker = st.selectbox("Smoker?", options=["Yes", "No"])
ethnicity = st.selectbox(
    "Ethnicity",
    options=["Mexican American", "Other Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Other/Multi-Racial"]
)

# --- Age warning ---
if age < 16 or age > 49:
    st.warning(
        "Predictions may be unreliable for participants under 16 or over 49 years old. "
        "The original NHANES study was limited to this age range."
    )

# --- Map categorical inputs ---
sex_val = 1 if sex == "Male" else 0
education_map = {"Less than high school": 0, "High school": 1, "Greater than high school": 2}
education_val = education_map[education_level]
smoker_map = {"Yes": "smoker_yes", "No": "smoker_no"}
smoker_feature = smoker_map[smoker]

ethnicity_map = {
    "Mexican American": "ethnic_mexican",
    "Other Hispanic": "ethnic_hispanic",
    "Non-Hispanic White": "ethnic_white",
    "Non-Hispanic Black": "ethnic_black",
    "Other/Multi-Racial": "ethnic_other",
}
ethnicity_feature = ethnicity_map[ethnicity]

# --- Correct feature order (29 features from training) ---
model_feature_order = [
    "age", "gender", "poverty_ratio", "family_income", "education_level",
    "health_insurance", "physical_activity_time", "systolic_bp",
    "diastolic_bp", "pulse_rate", "bmi", "waist_perimeter",
    "body_fat_percent", "weight_diff", "weight", "height", "cholesterol",
    "red_blood_cell_count", "hemoglobin", "hematocrit",
    "mean_cell_hemoglobin", "red_cell_distribution_width",
    "ethnic_mexican", "ethnic_hispanic", "ethnic_white",
    "ethnic_black", "ethnic_other",
    "smoker_former", "smoker_no", "smoker_yes",
]

# --- Build feature vector ---
features = []
for feat in model_feature_order:
    if feat == "age":
        val = age
    elif feat == "gender":
        val = sex_val
    elif feat == "body_fat_percent":
        val = body_fat_percent
    elif feat == "bmi":
        val = bmi
    elif feat == "education_level":
        val = education_val
    elif feat == smoker_feature:
        val = 1
    elif feat == ethnicity_feature:
        val = 1
    else:
        val = X_mean[feat]

    # --- Apply transformations ---
    if feat in log1p_features:
        val = np.log1p(val)
    elif feat in log_features:
        val = np.log(val)
    elif feat in sqrt_sym_features:
        val = np.sign(val) * np.sqrt(np.abs(val))

    features.append(val)

features = np.array(features).reshape(1, -1)
features_scaled = scaler.transform(features)

# --- Prediction and interpretation ---
def interpret_vo2max(vo2max, gender):
    thresholds = [51.4, 47.1, 41.4, 37.7] if gender == "Male" else [45.6, 39.9, 35.5, 32.1]
    if vo2max >= thresholds[0]:
        return "Superior"
    elif vo2max >= thresholds[1]:
        return "Excellent"
    elif vo2max >= thresholds[2]:
        return "Good"
    elif vo2max >= thresholds[3]:
        return "Fair"
    else:
        return "Poor"
        
if st.button("Predict VO₂max"):
    vo2max_log_pred = model.predict(features_scaled)[0]  # prediction in log scale
    vo2max_pred = np.exp(vo2max_log_pred)  # convert back to original VO2max scale
    st.success(f"Estimated VO₂max: {vo2max_pred:.2f} ml/kg/min")
    
    category = interpret_vo2max(vo2max_pred, sex)
    st.info(f"Your VO₂max category is: {category}")
