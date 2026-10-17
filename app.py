
import streamlit as st
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Predict heart disease using Machine Learning models.")
st.warning(
    "Educational project only. This prediction is not a medical diagnosis."
)

# Model folder
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Saved model filenames
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Decision Tree": "decision_tree.pkl",
    "SVM": "svm.pkl"
}

# Load models
@st.cache_resource
def load_model(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)

models = {}

for name, filename in MODEL_FILES.items():
    path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(path):
        try:
            models[name] = load_model(path)
        except Exception as e:
            st.error(f"Could not load {filename}: {e}")
    else:
        st.error(f"Model file not found: {path}")

st.divider()
st.subheader("Enter Patient Details")

# Patient input form
with st.form("heart_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", min_value=1, max_value=120, value=40
        )
        sex = st.selectbox("Sex", ["Male", "Female"])
        resting_bp = st.number_input(
            "Resting Blood Pressure", min_value=0,
            max_value=300, value=120
        )
        cholesterol = st.number_input(
            "Cholesterol", min_value=0,
            max_value=1000, value=200
        )
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl", [0, 1]
        )
        max_hr = st.number_input(
            "Maximum Heart Rate", min_value=0,
            max_value=250, value=150
        )
        oldpeak = st.number_input(
            "Oldpeak", min_value=-10.0,
            max_value=20.0, value=0.0, step=0.1
        )

    with col2:
        chest_pain = st.selectbox(
            "Chest Pain Type", ["ASY", "ATA", "NAP", "TA"]
        )
        resting_ecg = st.selectbox(
            "Resting ECG", ["Normal", "ST", "LVH"]
        )
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina", ["No", "Yes"]
        )
        st_slope = st.selectbox(
            "ST Slope", ["Down", "Flat", "Up"]
        )

    model_name = st.selectbox(
        "Choose ML Model",
        list(MODEL_FILES.keys())
    )

    submitted = st.form_submit_button(
        "Predict Heart Disease",
        use_container_width=True
    )

# Prediction
if submitted and models:

    # These columns must match the exact training features
    # and their order in X_train.
    input_data = pd.DataFrame([{
        "Age": float(age),
        "RestingBP": float(resting_bp),
        "Cholesterol": float(cholesterol),
        "FastingBS": int(fasting_bs),
        "MaxHR": float(max_hr),
        "Oldpeak": float(oldpeak),
        "Sex_M": sex == "Male",
        "ChestPainType_ATA": chest_pain == "ATA",
        "ChestPainType_NAP": chest_pain == "NAP",
        "ChestPainType_TA": chest_pain == "TA",
        "RestingECG_Normal": resting_ecg == "Normal",
        "RestingECG_ST": resting_ecg == "ST",
        "ExerciseAngina_Y": exercise_angina == "Yes",
        "ST_Slope_Flat": st_slope == "Flat",
        "ST_Slope_Up": st_slope == "Up"
    }])

    model = models.get(model_name)

    if model is None:
        st.error("Selected model could not be loaded.")
    else:
        try:
            prediction = model.predict(input_data)[0]

            st.subheader("Prediction Result")

            if int(prediction) == 1:
                st.error(
                    "The model predicts the Heart Disease class (1)."
                )
            else:
                st.success(
                    "The model predicts the No Heart Disease class (0)."
                )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                st.write("Model class probabilities:")
                st.write({
                    str(cls): f"{prob * 100:.2f}%"
                    for cls, prob in zip(
                        model.classes_, probabilities
                    )
                })

            st.caption(
                "This is a machine-learning classification result, "
                "not a diagnosis or a reliable estimate of personal risk."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                "Check that the input columns, encoding, and order "
                "match the features used to train the selected model."
            )

elif submitted:
    st.error("No models were loaded. Check your .pkl files.")
