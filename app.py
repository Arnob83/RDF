import streamlit as st
import sqlite3
import pickle
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
from shap.maskers import Independent

# Database for user authentication
def init_user_db():
    conn = sqlite3.connect("user_auth.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)
    conn.commit()
    conn.close()

# Add a default admin user if it doesn't exist
def add_default_user():
    conn = sqlite3.connect("user_auth.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", "admin123"))
        conn.commit()
    conn.close()

# Verify login credentials
def verify_user(username, password):
    conn = sqlite3.connect("user_auth.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

# Login function
def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if verify_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid username or password")

# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.success("You have been logged out.")

# Loan prediction logic
@st.cache_data
def prediction(Credit_History, Education, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, Property_Area, Gender):
    # Example logic for prediction
    # Replace this with your model and logic
    pred_label = "Approved" if Credit_History == "Clear Debts" else "Rejected"
    return pred_label

# Loan prediction app
def loan_prediction_app():
    st.title("Loan Prediction System")
    
    # Collect user inputs
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

    # Prediction button
    if st.button("Predict Loan Approval"):
        # Make the prediction
        result = prediction(
            Credit_History, Education, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, Property_Area, Gender
        )
        st.success(f"Prediction: **{result}**")

# Main function
def main():
    # Initialize the user database
    init_user_db()
    add_default_user()

    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

    # Check login state
    if st.session_state.logged_in:
        st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
        st.sidebar.button("Logout", on_click=logout)  # Logout button
        loan_prediction_app()  # Loan prediction app
    else:
        login()  # Login page

# Run the app
if __name__ == "__main__":
    main()
