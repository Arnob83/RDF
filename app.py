import sqlite3
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import requests
import shap

# URLs for the model and scaler files in your GitHub repository
model_url = "https://raw.githubusercontent.com/Arnob83/RDF/main/Logistic_Regression_model.pkl"
scaler_url = "https://raw.githubusercontent.com/Arnob83/RDF/main/scaler.pkl"
x_train_url = "https://raw.githubusercontent.com/Arnob83/RDF/main/X_train_scaled.pkl"

# Download the model and scaler files
def download_model_and_scaler():
    model_response = requests.get(model_url)
    with open("Logistic_Regression_model.pkl", "wb") as file:
        file.write(model_response.content)

    scaler_response = requests.get(scaler_url)
    with open("scaler.pkl", "wb") as file:
        file.write(scaler_response.content)

    x_train_response = requests.get(x_train_url)
    with open("X_train_scaled.pkl", "wb") as file:
        file.write(x_train_response.content)

    # Load model, scaler, and training data
    with open("Logistic_Regression_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with open("X_train_scaled.pkl", "rb") as x_train_file:
        x_train = pickle.load(x_train_file)
    return model, scaler, x_train

classifier, scaler, X_train_scaled = download_model_and_scaler()

# Initialize the SQLite database
def init_db():
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    
    # User table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone TEXT UNIQUE,
        password TEXT
    )
    """)
    
    # Predictions table
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
    if phone == "admin" and password == "admin123":  # Admin credentials
        return "admin"
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE phone = ? AND password = ?", (phone, password))
    user = cursor.fetchone()
    conn.close()
    return "user" if user else None

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

    # SHAP explainer
    explainer = shap.Explainer(classifier, X_train_scaled)
    shap_values = explainer(input_data_filtered)
    return pred_label, raw_input_data, input_data_filtered, probabilities, shap_values

# Main Streamlit app
def main():
    init_db()

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["user_role"] = None

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

        elif action == "Login":
            phone = st.text_input("Phone Number")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                user_role = authenticate_user(phone, password)
                if user_role == "user":
                    st.session_state["logged_in"] = True
                    st.session_state["user_role"] = "user"
                    st.success("Login successful!")
                elif user_role == "admin":
                    st.session_state["logged_in"] = True
                    st.session_state["user_role"] = "admin"
                    st.success("Admin login successful!")
                else:
                    st.error("Invalid phone number or password!")

    else:
        if st.session_state["user_role"] == "admin":
            st.header("Admin Panel")
            conn = sqlite3.connect("loan_data.db")
            df_predictions = pd.read_sql_query("SELECT * FROM loan_predictions", conn)
            conn.close()
            st.dataframe(df_predictions)
            st.download_button("Download Prediction Data", df_predictions.to_csv(index=False), "predictions.csv", "text/csv")
            if st.button("Logout"):
                st.session_state["logged_in"] = False
                st.success("Logged out successfully!")

        else:
            st.header("Loan Prediction")
            # Prediction inputs
            # (Code here for loan prediction form and SHAP explanation)
            if st.button("Logout"):
                st.session_state["logged_in"] = False
                st.success("Logged out successfully!")

if __name__ == "__main__":
    main()
