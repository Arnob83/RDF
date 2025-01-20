import sqlite3
import pickle
import streamlit as st
import requests
import os

# URLs for the model and scaler files in your GitHub repository
model_url = "https://raw.githubusercontent.com/Arnob83/RDF/main/Logistic_Regression_model.pkl"
scaler_url = "https://raw.githubusercontent.com/Arnob83/RDF/main/scaler.pkl"
x_train_url = "https://raw.githubusercontent.com/Arnob83/RDF/main/X_train_scaled.pkl"

# Download the model, scaler, and X_train files
def download_files():
    model_response = requests.get(model_url)
    with open("Logistic_Regression_model.pkl", "wb") as file:
        file.write(model_response.content)

    scaler_response = requests.get(scaler_url)
    with open("scaler.pkl", "wb") as file:
        file.write(scaler_response.content)

    response_x_train = requests.get(x_train_url)
    with open("X_train_scaled", "wb") as file:
        file.write(response_x_train.content)

    # Load the model, scaler, and X_train
    with open("Logistic_Regression_model.pkl", "rb") as model_file:
        classifier = pickle.load(model_file)

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    with open("X_train_scaled", "rb") as file:
        X_train_scaled = pickle.load(file)
    
    return classifier, scaler, X_train_scaled

# Initialize SQLite databases for users and loan data
def init_db():
    conn = sqlite3.connect("registration_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone_number TEXT UNIQUE,
        password TEXT
    )
    """)
    conn.commit()
    conn.close()

    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
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

# Save user registration data
def register_user(phone_number, password):
    conn = sqlite3.connect("registration_data.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (phone_number, password) VALUES (?, ?)", (phone_number, password))
    conn.commit()
    conn.close()

# Check if user exists
def check_user_exists(phone_number):
    conn = sqlite3.connect("registration_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE phone_number = ?", (phone_number,))
    user = cursor.fetchone()
    conn.close()
    return user

# Check user credentials
def check_credentials(phone_number, password):
    conn = sqlite3.connect("registration_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE phone_number = ? AND password = ?", (phone_number, password))
    user = cursor.fetchone()
    conn.close()
    return user

# Save loan prediction data
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

# Login function
def login():
    phone_number = st.text_input("Phone Number", key="login_phone_number")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        if phone_number == "admin" and password == "adminpassword":  # Admin credentials
            st.session_state["logged_in"] = True
            st.session_state["role"] = "admin"
            st.success("Logged in as Admin!")
        else:
            user = check_credentials(phone_number, password)
            if user:
                st.session_state["logged_in"] = True
                st.session_state["role"] = "user"
                st.success("Logged in successfully!")
            else:
                st.error("Invalid credentials or phone number not registered.")

# Register function
def register():
    phone_number = st.text_input("Phone Number", key="register_phone_number")
    password = st.text_input("Password", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")

    if st.button("Register"):
        if password != confirm_password:
            st.error("Passwords do not match.")
        elif check_user_exists(phone_number):
            st.error("User already registered with this phone number.")
        else:
            register_user(phone_number, password)
            st.success("Registration successful! You can now log in.")

# Logout function
def logout():
    st.session_state["logged_in"] = False
    st.session_state["role"] = None
    st.success("You have been logged out.")

# Main Streamlit app
def main():
    init_db()

    st.markdown(
        """
        <style>
        .main-container {
            background-color: #f4f6f9;
            border: 2px solid #e6e8eb;
            padding: 20px;
            border-radius: 10px;
        }
        .header {
            background-color: #4caf50;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .header h1 {
            color: white;
        }
        </style>
        <div class="main-container">
        <div class="header">
        <h1>Bank Loan Prediction ML App</h1>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["role"] = None

    if not st.session_state["logged_in"]:
        st.header("Login / Register")
        login()
        st.markdown("**Don't have an account?**")
        register()
    else:
        st.header("Please fill-up your personal information.")
        # Add your loan prediction form and logic here.

        st.divider()
        if st.button("Logout"):
            logout()

        if st.session_state["role"] == "admin":
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
        else:
            st.info("Only admins can download the database.")

if __name__ == "__main__":
    download_files()  # Download necessary files
    main()
