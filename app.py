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
model_url = "https://raw.githubusercontent.com/Arnob83/D-A/RDF/Random_Forest_model.pkl"
scaler_url = "https://raw.githubusercontent.com/Arnob83/RDF/main/scaler.pkl"

# Download the model and scaler files
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, "wb") as file:
        file.write(response.content)

# Download and load the model
model_path = "Random_Forest_model.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path):
    download_file(model_url, model_path)

if not os.path.exists(scaler_path):
    download_file(scaler_url, scaler_path)

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

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

# Prediction function
@st.cache_data
def prediction(input_data, _model, _scaler):
    # Apply feature scaling
    columns_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term']
    input_data[columns_to_scale] = _scaler.transform(input_data[columns_to_scale])

    # Ensure input data is ordered according to model feature order
    feature_order = _model.feature_names_in_
    input_data = input_data[feature_order]

    # Model prediction (0 = Rejected, 1 = Approved)
    prediction = _model.predict(input_data)
    pred_label = 'Approved' if prediction[0] == 1 else 'Rejected'
    return pred_label

# Explain prediction
def explain_prediction(input_data, model, final_result):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    shap_values_for_input = shap_values[0]

    feature_names = input_data.columns
    shap_values_for_plot = [
        shap_value[0] if isinstance(shap_value, np.ndarray) else shap_value
        for shap_value in shap_values_for_input
    ]

    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, shap_values_for_plot, color=["green" if val > 0 else "red" for val in shap_values_for_plot])
    plt.xlabel("SHAP Value (Impact on Prediction)")
    plt.ylabel("Features")
    plt.title("Feature Contributions to Prediction")
    plt.tight_layout()

    return plt

# Main Streamlit app
def main():
    init_db()

    st.title("Loan Prediction ML App")

    dependents_mapping = {'0': 0.6861, '1': 0.6471, '2': 0.7525, '3+': 0.6471}
    property_area_mapping = {'Rural': 0.6145, 'Semiurban': 0.7682, 'Urban': 0.6584}

    Gender = st.selectbox("Gender", ("Male", "Female"))
    Married = st.selectbox("Married", ("Yes", "No"))
    Dependents = st.selectbox("Dependents", ('0', '1', '2', '3+'))
    Self_Employed = st.selectbox("Self Employed", ("Yes", "No"))
    Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
    Property_Area = st.selectbox("Property Area", ("Urban", "Rural", "Semiurban"))
    Credit_History = st.selectbox("Credit History", ("Unclear Debts", "Clear Debts"))
    Education = st.selectbox('Education', ("Under_Graduate", "Graduate"))
    ApplicantIncome = st.number_input("Applicant's yearly Income", min_value=0.0)
    CoapplicantIncome = st.number_input("Co-applicant's yearly Income", min_value=0.0)
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0.0)

    if st.button("Predict"):
        Property_Area_encoded = property_area_mapping[Property_Area]
        Credit_History_encoded = 0 if Credit_History == "Unclear Debts" else 1
        Dependents_encoded = dependents_mapping[Dependents]

        input_data = pd.DataFrame([{
            "Credit_History": Credit_History_encoded,
            "Education": 1 if Education == "Graduate" else 0,
            "ApplicantIncome": ApplicantIncome,
            "CoapplicantIncome": CoapplicantIncome,
            "Loan_Amount_Term": Loan_Amount_Term,
            "Property_Area": Property_Area_encoded,
            "Dependents": Dependents_encoded
        }])

        result = prediction(input_data, model, scaler)

        save_to_database(Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area, 
                         Credit_History, Education, ApplicantIncome, CoapplicantIncome, 
                         Loan_Amount_Term, result)

        st.success(f"Loan Prediction: {result}")

        st.header("Explanation of Prediction")
        bar_chart = explain_prediction(input_data, model, final_result=result)
        st.pyplot(bar_chart)

if __name__ == '__main__':
    main()
