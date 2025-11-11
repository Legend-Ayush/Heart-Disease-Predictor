import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import About
import Result

#---Load and prepare the dataset---
heart = pd.read_csv("heart.csv")
X = heart.drop("target", axis=1)
y = heart["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101)
logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)
y_pred = logmodel.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
accuracy = report["accuracy"]

#---Streamlit UI Setup---
st.set_page_config(page_title="Heart Disease Predictor 💓", page_icon="❤", layout="wide")

st.sidebar.title("💓 Heart App Menu")

# ✅ Persist selected page
if "page" not in st.session_state:
    st.session_state.page = "🏠 Predictor"

page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Predictor","📄 Result","💬 About"],
    index=["🏠 Predictor","📄 Result","💬 About"].index(st.session_state.page)
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Built by Ayush Singh 💻\n\n"
    "Using Streamlit & Logistic Regression"
)

#--------- PREDICTOR PAGE ---------
if page == "🏠 Predictor":
    st.title("🏥 Heart Disease Prediction App")
    st.markdown("## 🩺 Heart Disease Risk Predictor")

    st.info("""
This app predicts the likelihood of heart disease using a trained Machine Learning model.
""")

    st.subheader("Enter your details below:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 40)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col2:
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 60, 250, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
        ca = st.selectbox("No. of Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalium Stress Test Result", [0, 1, 2, 3])

    # prepare input dataframe
    user_data = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "cp": [cp],
            "trestbps": [trestbps],
            "chol": [chol],
            "fbs": [fbs],
            "restecg": [restecg],
            "thalach": [thalach],
            "exang": [exang],
            "oldpeak": [oldpeak],
            "slope": [slope],
            "ca": [ca],
            "thal": [thal],
        }
    )

    # RUN PREDICTION
    if st.button("🔍 Predict"):
        prediction = logmodel.predict(user_data)[0]
        probability = logmodel.predict_proba(user_data)[0][1]

        # ✅ store only input for Result page
        st.session_state['last_input'] = user_data

        st.markdown("---")
        st.subheader("🧠 Model Prediction Result")

        if prediction == 1:
            st.error(f"⚠ The model predicts **presence of heart disease** with **{probability*100:.2f}%** confidence.")
        else:
            st.success(f"✅ The model predicts **no heart disease** with **{(1-probability)*100:.2f}%** confidence.")

        st.progress(int(probability * 100))
        st.caption(f"Model tested accuracy: *{accuracy:.2%}*")

    # KNOW YOUR RESULT BUTTON (right side)
    colL, colR = st.columns([3, 1])
    with colR:
        if st.button("➡ Know Your Result"):
            if "last_input" in st.session_state:
                st.session_state.page = "📄 Result"
                st.rerun()
            else:
                st.warning("⚠ Please Predict first!")


#--------- RESULT PAGE ---------
elif page == "📄 Result":
    Result.run()

#--------- ABOUT PAGE ---------
elif page == "💬 About":
    About.run()
