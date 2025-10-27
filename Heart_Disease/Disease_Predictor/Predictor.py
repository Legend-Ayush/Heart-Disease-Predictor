import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Load and prepare the dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "heart.csv")
heart = pd.read_csv("heart.csv")
X = heart.drop("target", axis=1)
y = heart["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101
)

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)

y_pred = logmodel.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = report["accuracy"]


# Streamlit UI Setup
st.set_page_config(page_title="Heart Disease Predictor 💓", page_icon="❤️", layout="wide")

st.sidebar.title("💓 Heart App Menu")
page = st.sidebar.radio("Navigate to:", ["🏠 Predictor","💬 About"])

st.sidebar.markdown("---")
st.sidebar.info(
    "Built by Ayush Singh 💻\n\n"
    "Using Streamlit & Logistic Regression ❤️"
)


# PAGE 1: Predictor
if page == "🏠 Predictor":
    st.title("🏥 Heart Disease Prediction App")
    st.markdown("## 🩺 Heart Disease Risk Predictor")

    st.info("""
This app helps you **predict the likelihood of heart disease** based on your personal health data.  
It uses a trained **Machine Learning model** to analyze your inputs and estimate your risk level.
""")

    st.markdown("""
### 🧾 How to use:
1. Fill in your health details such as **age, gender, blood pressure, cholesterol**, **heart rate**, **etc**.  
2. Click on the **Predict** button at the bottom.  
3. The app will instantly display your **predicted risk result**.
""")

    st.warning("""
⚠️ **Disclaimer:**  
This tool is for **educational and informational purposes only**.  
It is **not a substitute** for professional medical advice or diagnosis.
""")
    
    st.markdown("---")

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

    if st.button("🔍 Predict"):
        prediction = logmodel.predict(user_data)[0]
        probability = logmodel.predict_proba(user_data)[0][1]

        st.markdown("---")
        st.subheader("🧭 Prediction Confidence")
        st.progress(int(probability * 100))

        if prediction == 1:
            st.error(f"⚠️ The model predicts **presence** of heart disease ({probability*100:.2f}% confidence).")
        else:
            st.success(f"✅ The model predicts **no heart disease** ({(1 - probability)*100:.2f}% confidence).")

        st.caption(f"Model tested accuracy: **{accuracy:.2%}**")


# PAGE 2: About
elif page == "💬 About":
    st.title("💖 About This App")
    st.markdown(
            """
    ## 🩺 Heart Disease Prediction App

    This web application predicts the **likelihood of heart disease** based on user-provided medical data.  
    It uses a **Logistic Regression** machine learning model trained on the **UCI Heart Disease Dataset**.

    ---
    ### 📘 Project Overview

    Cardiovascular disease remains one of the leading causes of mortality worldwide.  
    Early prediction can help people make informed lifestyle and medical decisions.  

    The **Heart Disease Predictor** simplifies this process by:
    - Collecting 13 essential medical inputs (e.g., age, cholesterol, blood pressure)
    - Feeding them into a trained Logistic Regression model
    - Returning the probability of heart disease presence or absence

    ---
    ### ⚙️ How It Works

    1. **User Input:**  
       You provide key health indicators like age, cholesterol, fasting blood sugar, and more.

    2. **Feature Processing:**  
       Inputs are normalized and aligned to the model’s training features.

    3. **Prediction:**  
       The trained Logistic Regression model outputs:
       - **1** → Presence of heart disease  
       - **0** → Absence of heart disease  
       along with a confidence probability.

    4. **Results Display:**  
       - A progress bar shows prediction confidence  
       - Clear text result summarizing your health prediction

    ---
    ### 🧠 Machine Learning Model

    **Algorithm:** Logistic Regression  
    **Library Used:** scikit-learn  
    **Dataset:** UCI Heart Disease Dataset  
    **Training-Test Split:** 60% - 40%  
    **Accuracy:** ~83–85% on test data  

    The model was chosen for:
    - Interpretability (you can see feature impacts)
    - Low computational cost
    - High baseline performance on medical classification problems

    ---
    ### 📊 Dataset Features

    | Feature | Description |
    |----------|-------------|
    | **age** | Age in years |
    | **sex** | 1 = Male, 0 = Female |
    | **cp** | Chest pain type (0–3) |
    | **trestbps** | Resting blood pressure (mm Hg) |
    | **chol** | Serum cholesterol (mg/dl) |
    | **fbs** | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
    | **restecg** | Resting ECG results (0–2) |
    | **thalach** | Maximum heart rate achieved |
    | **exang** | Exercise-induced angina (1 = yes; 0 = no) |
    | **oldpeak** | ST depression induced by exercise relative to rest |
    | **slope** | Slope of peak exercise ST segment (0–2) |
    | **ca** | Number of major vessels (0–3) colored by fluoroscopy |
    | **thal** | Thalium stress test result (0–3) |

    ---
    ### 💾 Technologies Used

    - **Python 3.12+**
    - **Pandas / NumPy** for data handling  
    - **scikit-learn** for training & prediction  
    - **Streamlit** for web interface  

    ---
    ### 💬 Author

    **Developed by:** Ayush Singh  
    **GitHub:** [github.com/AyushSingh]https://github.com/Legend-Ayush  
    **Framework:** Streamlit  
    **Model:** Logistic Regression  

    ---
    ### ❤️ Future Improvements

    - Add advanced ML models (Random Forest, XGBoost)
    - Include visualization for feature importance
    - Enable real-time data input via wearable devices
    - Cloud deployment for public access

    ---
    ### 📚 Disclaimer

    This app is intended for **educational and research purposes only**.  
    It is **not a medical diagnostic tool** and should not replace professional medical advice.

    ---
    **Thank you for exploring! 💓**
    """
    )
