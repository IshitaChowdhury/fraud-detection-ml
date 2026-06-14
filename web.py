
import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="AI-Powered Fraud Detection System",
    layout="wide"
)

# Load Model
model = joblib.load("fraud_detection_pipeline.pkl")

# Header
st.title("AI-Powered Fraud Detection System")
st.markdown(
    "Predict whether a financial transaction is fraudulent using a Machine Learning model."
)

st.divider()

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        """
        This application analyzes transaction details and predicts
        whether the transaction is potentially fraudulent.

        Model: Machine Learning Pipeline
        """
    )

# Input Section
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0
    )

    oldbalanceOrg = st.number_input(
        "Sender Old Balance",
        min_value=0.0,
        value=1000.0
    )

with col2:
    newbalanceOrig = st.number_input(
        "Sender New Balance",
        min_value=0.0,
        value=900.0
    )

    oldbalanceDest = st.number_input(
        "Receiver Old Balance",
        min_value=0.0,
        value=0.0
    )

    newbalanceDest = st.number_input(
        "Receiver New Balance",
        min_value=0.0,
        value=0.0
    )

st.divider()

# Transaction Summary
with st.expander("Transaction Summary"):
    st.write(f"Transaction Type: {transaction_type}")
    st.write(f"Transaction Amount: ₹{amount:,.2f}")
    st.write(f"Sender Balance: ₹{oldbalanceOrg:,.2f} → ₹{newbalanceOrig:,.2f}")
    st.write(f"Receiver Balance: ₹{oldbalanceDest:,.2f} → ₹{newbalanceDest:,.2f}")

# Prediction
if st.button("Analyze Transaction", use_container_width=True):

    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "Potential Fraud Detected. This transaction may require additional verification."
        )
    else:
        st.success(
            "Transaction appears legitimate based on the provided information."
        )

    st.metric(
        label="Prediction",
        value="Fraud" if prediction == 1 else "Legitimate"
    )

# Footer
st.divider()
st.caption("Built with Streamlit and Scikit-Learn")

