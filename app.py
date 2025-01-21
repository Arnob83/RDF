import sqlite3
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
import shap
from shap.maskers import Independent
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

if not openai.api_key:
    st.error("OpenAI API key not found in secrets. Please set it in the Streamlit Cloud Secrets.")

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

# Initialize SQLite database
def init_db():
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

# Text generation function (Updated to use gpt-3.5-turbo or gpt-4)
def generate_explanation_and_suggestions(features, shap_values, final_result):
    if final_result == "Rejected":
        base_prompt = f"""
        A user has applied for a loan, but their application was rejected. 
        Here are the SHAP feature contributions to the rejection:

        Features: {features}
        SHAP Values: {shap_values}

        Based on the above, explain to the user why their loan was rejected. 
        Also, provide actionable suggestions to improve their chances of loan approval in the future.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Updated model
                messages=[
                    {"role": "system", "content": "You are an assistant that explains loan rejections and provides actionable suggestions."},
                    {"role": "user", "content": base_prompt},
                ],
                max_tokens=300,
                temperature=0.7
            )
            explanation_and_suggestions = response["choices"][0]["message"]["content"].strip()
        except openai.error.AuthenticationError:
            explanation_and_suggestions = "Authentication error with OpenAI API. Please check your API key."
        except openai.error.RateLimitError:
            explanation_and_suggestions = "Rate limit exceeded. Please try again later."
        except openai.error.OpenAIError as e:
            explanation_and_suggestions = f"An error occurred with OpenAI API: {str(e)}"
    else:
        explanation_and_suggestions = "Congratulations! Your loan application was approved. Keep up the good financial behavior."

    return explanation_and_suggestions

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

# Explanation function
def explain_prediction(input_data_filtered, final_result):
    masker = Independent(X_train_scaled)
    explainer = shap.LinearExplainer(classifier, masker)
    shap_values = explainer.shap_values(input_data_filtered)
    shap_values_for_input = shap_values[0]

    feature_names = input_data_filtered.columns
    features_with_values = {feature: round(value, 2) for feature, value in zip(feature_names, shap_values_for_input)}

    explanation_text = f"**Why your loan is {final_result}:**\n\n"
    for feature, shap_value in zip(feature_names, shap_values_for_input):
        explanation_text += (
            f"- **{feature}**: {'Positive' if shap_value > 0 else 'Negative'} contribution with a SHAP value of {shap_value:.2f}\n"
        )

    # Generate suggestions using the updated text generation model
    suggestions = generate_explanation_and_suggestions(features_with_values, shap_values_for_input, final_result)

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

    return explanation_text, suggestions, plt

# Login function
def login():
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "password":  # Replace with your credentials
            st.session_state["logged_in"] = True
            st.session_state["role"] = "admin"
            st.success("Logged in as Admin! Please press login button.")
        elif username == "user" and password == "password":  # Replace with user credentials
            st.session_state["logged_in"] = True
            st.session_state["role"] = "user"
            st.success("Logged in as User! Please press login button.")
        else:
            st.error("Invalid credentials")

# Logout function
def logout():
    st.session_state["logged_in"] = False
    st.session_state["role"] = None
    st.success("Logged out successfully")

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
        st.header("Login")
        login()
    else:
        st.header("Please fill-up your personal information.")

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
                st.write("Probabilities (Rejected: 0, Approved: 1):", probabilities)

                explanation_text, suggestions, shap_plot = explain_prediction(processed_input, result)
                st.markdown(explanation_text)
                st.pyplot(shap_plot)

                # Display text generation output
                if result == "Rejected":
                    st.markdown("### Suggestions for Improving Loan Approval Chances")
                    st.write(suggestions)

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
    main()
