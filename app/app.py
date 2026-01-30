import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

model = joblib.load("model/churn_model.pkl")
training_columns = joblib.load("model/training_columns.pkl")

st.title("📉 Customer Churn Prediction App")
st.write("Enter customer details and predict churn probability.")

st.markdown("---")

tenure = st.slider("Tenure (months)", 0, 72, 10)
monthly_charges = st.number_input("Monthly Charges", value=70.0)
total_charges = st.number_input("Total Charges", value=700.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service,
    "PaymentMethod": payment_method
}

input_df = pd.DataFrame([input_data])
input_df_encoded = pd.get_dummies(input_df)

for col in training_columns:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

input_df_encoded = input_df_encoded[training_columns]

if st.button("Predict"):
    pred = model.predict(input_df_encoded)[0]
    prob = model.predict_proba(input_df_encoded)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer will CHURN (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer will NOT churn (Probability: {prob:.2f})")

st.markdown("---")
st.caption("Developed by Vedant Gadage")
