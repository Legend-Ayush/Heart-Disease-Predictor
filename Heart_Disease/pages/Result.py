import streamlit as st

def run():
    st.title("📄 Diagnosis Report")

    # ✅ Check if user has predicted
    if "last_input" not in st.session_state:
        st.warning("⚠ Please go back to Predictor, fill your health details and click Predict first.")
        return

    user_data = st.session_state["last_input"]
    row = user_data.iloc[0]   # extract first row as series

    st.subheader("🧾 Your Provided Details")
    st.dataframe(user_data)

    # ---------------- HEALTH ANALYSIS ----------------
    st.subheader("🩺 Analysis Based on Your Inputs:")

    warnings = []

    if row["cp"] in [2, 3]:
        warnings.append("⚠️ Chest pain type indicates possible heart disease risk.")

    if row["trestbps"] > 130:
        warnings.append(f"⚠️ High resting blood pressure ({row['trestbps']} mmHg).")

    if row["chol"] > 240:
        warnings.append(f"⚠️ High cholesterol level ({row['chol']} mg/dl).")

    if row["fbs"] == 1:
        warnings.append("⚠️ High fasting blood sugar detected.")

    if row["restecg"] == 2:
        warnings.append("⚠️ ECG shows abnormal patterns.")

    if row["thalach"] < 120:
        warnings.append(f"⚠️ Low maximum heart rate ({row['thalach']}).")

    if row["exang"] == 1:
        warnings.append("⚠️ Exercise-induced angina detected.")

    if row["oldpeak"] > 2.0:
        warnings.append(f"⚠️ High ST depression ({row['oldpeak']}).")

    if row["slope"] == 2:
        warnings.append("⚠️ Downsloping ST segment — risky pattern.")

    if row["ca"] > 1:
        warnings.append(f"⚠️ Blockage in major coronary vessels ({row['ca']}).")

    if row["thal"] in [2, 3]:
        warnings.append("⚠️ Thalium stress test indicates abnormal result.")

    # ----------- OUTPUT -------------
    if len(warnings) == 0:
        st.success("No concerning indicators found. ❤️")
    else:
        for w in warnings:
            st.error(w)
        st.warning("🩺 Recommendation: Please consult a cardiologist for further checkup.")

    # ----------- BACK BUTTON -------------
    if st.button("🔙 Check Again"):
        st.session_state["page"] = "🏠 Predictor"
        st.rerun()


# ✅ Required to run when this file loads in Streamlit pages mode
if __name__ == "__main__":
    run()
