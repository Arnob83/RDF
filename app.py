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
# Updated prediction function
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

# Explanation function
def explain_prediction(input_data, final_result):
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(input_data)
    shap_values_for_input = shap_values[0]

    feature_names = input_data.columns
    explanation_text = f"**Why your loan is {final_result}:**\n\n"
    for feature, shap_value in zip(feature_names, shap_values_for_input):
        explanation_text += (
            f"- **{feature}**: {'Positive' if shap_value > 0 else 'Negative'} contribution with a SHAP value of {shap_value:.2f}\n"
        )
    if final_result == 'Rejected':
        explanation_text += "\nThe loan was rejected because the negative contributions outweighed the positive ones."
    else:
        explanation_text += "\nThe loan was approved because the positive contributions outweighed the negative ones."

    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, shap_values_for_input, color=["green" if val > 0 else "red" for val in shap_values_for_input])
    plt.xlabel("SHAP Value (Impact on Prediction)")
    plt.ylabel("Features")
    plt.title("Feature Contributions to Prediction")
    plt.tight_layout()
    return explanation_text, plt


# Main Streamlit app
def main():
    # Initialize database
    init_db()

    # App layout
    st.title("Loan Prediction ML App")

    # Define mappings for target-encoded features
    dependents_mapping = {'0': 0.6861, '1': 0.6471, '2': 0.7525, '3+': 0.6471}
    property_area_mapping = {'Rural': 0.6145, 'Semiurban': 0.7682, 'Urban': 0.6584}

    # User inputs
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
        # Map inputs to encoded values
        Property_Area_encoded = property_area_mapping[Property_Area]
        Credit_History_encoded = 0.0 if Credit_History == "Unclear Debts" else 1.0
        Dependents_encoded = dependents_mapping[Dependents]

        # Prepare input data for prediction
        input_data = pd.DataFrame([{
            "Credit_History": Credit_History_encoded,
            "Education": 1 if Education == "Graduate" else 0,
            "ApplicantIncome": ApplicantIncome,
            "CoapplicantIncome": CoapplicantIncome,
            "Loan_Amount_Term": Loan_Amount_Term,
            "Property_Area": Property_Area_encoded,
            "Dependents": Dependents_encoded
        }])

        # Prediction
        result = prediction(input_data, model, scaler)

        # Save to database
        save_to_database(Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area, 
                         Credit_History, Education, ApplicantIncome, CoapplicantIncome, 
                         Loan_Amount_Term, result)

        # Display result
        st.success(f"Loan Prediction: {result}")

      # Explain the prediction
        st.header("Explanation of Prediction")
        explanation_text, bar_chart = explain_prediction(input_data, final_result=result)
        st.write(explanation_text)
        st.pyplot(bar_chart)


    # View database
    if st.button("View Database"):
        conn = sqlite3.connect("loan_data.db")
        df = pd.read_sql_query("SELECT * FROM loan_predictions", conn)
        st.write(df)
        conn.close()

    # Download database button
    if st.button("Download Database"):
        if os.path.exists("loan_data.db"):
            with open("loan_data.db", "rb") as f:
                st.download_button(
                    label="Download SQLite Database",
                    data=f,
                    file_name="loan_data.db",
                    mime="application/octet-stream"
                )
        else:
            st.error("Database file not found.")

if __name__ == '__main__':
    main()
