import sqlite3
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
import shap
from shap.maskers import Independent

# URLs for the model and scaler files in your GitHub repository
model_url = "https://raw.githubusercontent.com/Arnob83/RDF/main/Logistic_Regression_model.pkl"
scaler_url = "https://raw.githubusercontent.com/Arnob83/RDF/main/scaler.pkl"
x_train_url = "https://raw.githubusercontent.com/Arnob83/RDF/main/X_train_scaled.pkl"

# Download the model file and save it locally
model_response = requests.get(model_url)
with open("Logistic_Regression_model.pkl", "wb") as file:
    file.write(model_response.content)

# Download the scaler file and save it locally
scaler_response = requests.get(scaler_url)
with open("scaler.pkl", "wb") as file:
    file.write(scaler_response.content)

# Load the trained model
with open("Logistic_Regression_model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

# Load the scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Download and save the X_train.pkl file
response_x_train = requests.get(x_train_url)
with open("X_train_scaled", "wb") as file:
    file.write(response_x_train.content)

# Load X_train
with open("X_train_scaled", "rb") as file:
    X_train_scaled = pickle.load(file)

# Initialize SQLite database for users and loan data
def init_db():
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    
    # User table for login and registration
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone TEXT UNIQUE,
        password TEXT
    )
    """)
    
    # Loan predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loan_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name TEXT,
        gender TEXT,
        married TEXT,
        dependents INTEGER,
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

# Register a new user
def register_user(phone, password):
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (phone, password) VALUES (?, ?)", (phone, password))
        conn.commit()
        st.success("Registration successful! Please log in.")
    except sqlite3.IntegrityError:
        st.error("Phone number already registered!")
    conn.close()

# Authenticate user
def authenticate_user(phone, password):
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE phone = ? AND password = ?", (phone, password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

# Save prediction data to the database
def save_to_database(customer_name, gender, married, dependents, self_employed, loan_amount, property_area, 
                     credit_history, education, applicant_income, coapplicant_income, 
                     loan_amount_term, result):
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO loan_predictions (
        customer_name, gender, married, dependents, self_employed, loan_amount, property_area, 
        credit_history, education, applicant_income, coapplicant_income, loan_amount_term, result
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (customer_name, gender, married, dependents, self_employed, loan_amount, property_area, 
          credit_history, education, applicant_income, coapplicant_income, 
          loan_amount_term, result))
    conn.commit()
    conn.close()

# Prediction function
@st.cache_data
def prediction(Credit_History, Education, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, Property_Area, Gender):
    Education = 0 if Education == "Graduate" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1
    
    property_area_mapping = {'Rural': 0.6145, 'Semiurban': 0.7682, 'Urban': 0.6584}
    Property_Area = property_area_mapping.get(Property_Area, 0.6145)

    Gender = 1 if Gender == "Male" else 0

    input_data = pd.DataFrame(
        [[Credit_History, Education, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, Property_Area, Gender]],
        columns=["Credit_History", "Education", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term", "Property_Area", "Gender"]
    )

    raw_input_data = input_data.copy()

    features_to_scale = input_data[["ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]]
    scaled_features = scaler.transform(features_to_scale)
    input_data[["ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]] = scaled_features

    trained_features = classifier.feature_names_in_
    input_data_filtered = input_data[trained_features]

    prediction = classifier.predict(input_data_filtered)
    probabilities = classifier.predict_proba(input_data_filtered)
    
    pred_label = 'Approved' if prediction[0] == 1 else 'Rejected'
    return pred_label, raw_input_data, input_data_filtered, probabilities

# Main Streamlit app
def main():
    init_db()

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["user_phone"] = None

    if not st.session_state["logged_in"]:
        st.header("Login or Register")

        action = st.radio("Choose an action", ["Login", "Register"])

        if action == "Register":
            phone = st.text_input("Phone Number")
            password = st.text_input("Password", type="password")

            if st.button("Register"):
                if phone and password:
                    register_user(phone, password)
                else:
                    st.error("Please fill in all fields!")

        if action == "Login":
            phone = st.text_input("Phone Number")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if authenticate_user(phone, password):
                    st.session_state["logged_in"] = True
                    st.session_state["user_phone"] = phone
                    st.success("Login successful!")
                else:
                    st.error("Invalid phone number or password!")

    else:
        st.header("Loan Prediction")
        Customer_Name = st.text_input("Customer Name")
        Gender = st.selectbox("Gender", ("Male", "Female"))
        Married = st.selectbox("Married", ("Yes", "No"))
        Dependents = st.selectbox("Dependents", (0, 1, 2, 3, 4, 5))
        Self_Employed = st.selectbox("Self Employed", ("Yes", "No"))
        Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
        Property_Area = st.selectbox("Property Area", ("Urban", "Rural", "Semi-urban"))
        Credit_History = st.selectbox("Credit History", ("Unclear Debts", "Clear Debts"))
        Education = st.selectbox('Education', ("Under_Graduate", "Graduate"))
        ApplicantIncome = st.number_input("Applicant's yearly Income", min_value=0.0)
        CoapplicantIncome = st.number_input("Co-applicant's yearly Income", min_value=0.0)
        Loan_Amount_Term = st.number_input("Loan Amount Term (in months)", min_value=0)

        if st.button("Predict Loan Approval"):
            if not Customer_Name:
                st.error("Please enter the customer's name.")
            else:
                result, raw_input, processed_input, probabilities = prediction(
                    Credit_History, Education, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, Property_Area, Gender
                )
                save_to_database(Customer_Name, Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area, 
                                 Credit_History, Education, ApplicantIncome, CoapplicantIncome, 
                                 Loan_Amount_Term, result)
                st.success(f"Prediction: **{result}**")
                st.write("Probabilities:", probabilities)

        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["user_phone"] = None
            st.success("Logged out successfully!")

if __name__ == "__main__":
    main()
