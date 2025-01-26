import streamlit as st
import pandas as pd
from keras.models import load_model
import pickle

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

model = load_model('customer_churn_model.keras')

st.title("Customer Churn Prediction App")
st.write("Predict whether a customer is likely to 'Exit' or 'Stay'.")

st.sidebar.header("Input Customer Details")

def user_input_features():
    Gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    Geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
    CreditScore = st.sidebar.slider("Credit Score", 300, 900, 600)
    Age = st.sidebar.slider("Age", 18, 100, 35)
    Tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
    Balance = st.sidebar.number_input("Balance (in $)", 0, 250000, 50000)
    NumOfProducts = st.sidebar.slider("Number of Products", 1, 4, 2)
    HasCrCard = st.sidebar.selectbox("Has Credit Card?", [1, 0])
    IsActiveMember = st.sidebar.selectbox("Is Active Member?", [1, 0])
    EstimatedSalary = st.sidebar.number_input("Estimated Salary (in $)", 0, 200000, 70000)

    data = {
        "Gender": [Gender],
        "Geography": [Geography],
        "CreditScore": [CreditScore],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance],
        "NumOfProducts": [NumOfProducts],
        "HasCrCard": [HasCrCard],
        "IsActiveMember": [IsActiveMember],
        "EstimatedSalary": [EstimatedSalary]
    }
    return pd.DataFrame(data)

input_data = user_input_features()

if st.button("Predict"):
    st.subheader("Customer Details")
    st.write(input_data)

    input_data_transformed = preprocessor.transform(input_data)

    prediction_probs = model.predict(input_data_transformed)
    prediction = (prediction_probs > 0.5).astype(int)

    st.subheader("Prediction Result")
    if prediction[0][0] == 1:
        st.error("The customer is likely to EXIT.")
    else:
        st.success("The customer is likely to STAY.")

