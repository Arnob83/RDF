import sqlite3
import pickle
import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import os

# URLs for the model and scaler files
model_url = "https://raw.githubusercontent.com/Arnob83/RDF/main/Random_Forest_model.pkl"
scaler_url = "https://raw.githubusercontent.com/Arnob83/RDF/main/scaler.pkl"

# Download the model file and save it locally
model_response = requests.get(model_url)
with open("Random_Forest_model.pkl", "wb") as file:
    file.write(model_response.content)

# Download the scaler file and save it locally
scaler_response = requests.get(scaler_url)
with open("scaler.pkl", "wb") as file:
    file.write(scaler_response.content)

# Load the trained model
with open("Random_Forest_model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

# Load the scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Encoding mappings
dependents_mapping = {'0': 0.6861, '1': 0.6471, '2': 0.7525, '3+': 0.6471}
property_area_mapping = {'Rural': 0.6145, 'Semiurban': 0.7682, 'Urban': 0.6584}

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loan_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gender TEXT,
        married TEXT,
        dependents TEXT,
        self_employed TEXT,
        loan_amount REAL,
        property_area TEXT,
        credit_history TEXT,
        education TEXT,
        applicant_income REAL,
        coapplicant_income REAL,
        loan_amount_term REAL,
        result TEXT
    )
    """)
    conn.commit()
    conn.close()

# Save prediction data to the database
def save_to_database(gender, married, dependents, self_employed, loan_amount, property_area, 
                     credit_history, education, applicant_income, coapplicant_income, 
                     loan_amount_term, result):
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO loan_predictions (
        gender, married, dependents, self_employed, loan_amount, property_area, 
        credit_history, education, applicant_income, coapplicant_income, loan_amount_term, result
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (gender, married, dependents, self_employed, loan_amount, property_area, 
          credit_history, education, applicant_income, coapplicant_income, 
          loan_amount_term, result))
    conn.commit()
    conn.close()

@st.cache_data
def prediction(Credit_History, Education, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, Dependents, Property_Area):
    # Map inputs using the mappings
    Dependents = dependents_mapping[Dependents]
    Property_Area = property_area_mapping[Property_Area]

    # Create input data for prediction
    input_data = pd.DataFrame([{
        "Credit_History": Credit_History,
        "Education": 1 if Education == "Graduate" else 0,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Dependents": Dependents,
        "Property_Area": Property_Area
    }])

    # Predict using the classifier
    prediction = classifier.predict(input_data)
    pred_label = 'Approved' if prediction[0] == 1 else 'Rejected'
    return pred_label, input_data

# Main Streamlit app
def main():
    init_db()

    # App layout
    st.title("Loan Prediction ML App")

    # User inputs
    Credit_History = st.selectbox("Credit History", ("Unclear Debts", "Clear Debts"))
    Gender = st.selectbox("Gender", ("Male", "Female"))
    Married = st.selectbox("Married", ("Yes", "No"))
    Dependents = st.selectbox("Dependents", ('0', '1', '2', '3+'))
    Self_Employed = st.selectbox("Self Employed", ("Yes", "No"))
    Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
    Property_Area = st.selectbox("Property Area", ("Urban", "Rural", "Semiurban"))
    Education = st.selectbox("Education", ("Under_Graduate", "Graduate"))
    ApplicantIncome = st.number_input("Applicant's yearly Income", min_value=0.0)
    CoapplicantIncome = st.number_input("Co-applicant's yearly Income", min_value=0.0)
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0.0)

    if st.button("Predict"):
        # Convert Credit_History to numeric
        Credit_History = 1 if Credit_History == "Clear Debts" else 0

        # Make prediction
        result, input_data = prediction(
            Credit_History,
            Education,
            ApplicantIncome,
            CoapplicantIncome,
            Loan_Amount_Term,
            Dependents,
            Property_Area
        )

        # Save data to database
        save_to_database(Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area, 
                         Credit_History, Education, ApplicantIncome, CoapplicantIncome, 
                         Loan_Amount_Term, result)

        # Display prediction result
        st.success(f"Your loan is {result}.")

if __name__ == '__main__':
    main()
