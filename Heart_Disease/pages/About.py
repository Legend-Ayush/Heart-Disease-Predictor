import streamlit as st 

def run():
   st.title("💖 About This App")
   st.markdown(
            """
    ## 🩺 Heart Disease Prediction App

    This web application predicts the *likelihood of heart disease* based on user-provided medical data.  
    It uses a *Logistic Regression* machine learning model trained on the *UCI Heart Disease Dataset*.

    ---
    ### 📘 Project Overview

    Cardiovascular disease remains one of the leading causes of mortality worldwide.  
    Early prediction can help people make informed lifestyle and medical decisions.  

    The *Heart Disease Predictor* simplifies this process by:
    - Collecting 13 essential medical inputs (e.g., age, cholesterol, blood pressure)
    - Feeding them into a trained Logistic Regression model
    - Returning the probability of heart disease presence or absence

    ---
    ### ⚙ How It Works

    1. *User Input:*  
       You provide key health indicators like age, cholesterol, fasting blood sugar, and more.

    2. *Feature Processing:*  
       Inputs are normalized and aligned to the model’s training features.

    3. *Prediction:*  
       The trained Logistic Regression model outputs:
       - *1* → Presence of heart disease  
       - *0* → Absence of heart disease  
       along with a confidence probability.

    4. *Results Display:*  
       - A progress bar shows prediction confidence  
       - Clear text result summarizing your health prediction

    ---
    ### 🧠 Machine Learning Model

    *Algorithm:* Logistic Regression  
    *Library Used:* scikit-learn  
    *Dataset:* UCI Heart Disease Dataset  
    *Training-Test Split:* 60% - 40%  
    *Accuracy:* ~83–85% on test data  

    The model was chosen for:
    - Interpretability (you can see feature impacts)
    - Low computational cost
    - High baseline performance on medical classification problems

    ---
    ### 📊 Dataset Features

    | Feature | Description |
    |----------|-------------|
    | *age* | Age in years |
    | *sex* | 1 = Male, 0 = Female |
    | *cp* | Chest pain type (0–3) |
    | *trestbps* | Resting blood pressure (mm Hg) |
    | *chol* | Serum cholesterol (mg/dl) |
    | *fbs* | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
    | *restecg* | Resting ECG results (0–2) |
    | *thalach* | Maximum heart rate achieved |
    | *exang* | Exercise-induced angina (1 = yes; 0 = no) |
    | *oldpeak* | ST depression induced by exercise relative to rest |
    | *slope* | Slope of peak exercise ST segment (0–2) |
    | *ca* | Number of major vessels (0–3) colored by fluoroscopy |
    | *thal* | Thalium stress test result (0–3) |

    ---
    ### 💾 Technologies Used

    - *Python 3.12+*
    - *Pandas / NumPy* for data handling  
    - *scikit-learn* for training & prediction  
    - *Streamlit* for web interface  

    ---
    ### 💬 Author

    *Developed by:* Ayush Singh  
    *GitHub:* [github.com/AyushSingh]https://github.com/Legend-Ayush  
    *Framework:* Streamlit  
    *Model:* Logistic Regression  

    ---
    ### 📚 Disclaimer

    This app is intended for *educational and research purposes only*.  
    It is *not a medical diagnostic tool* and should not replace professional medical advice.

    ---
    *Thank you for exploring! 💓*
    """
    )