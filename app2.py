import streamlit as st  # type: ignore
import numpy as np  # type: ignore
import pickle
from PIL import Image   # type: ignore  

# Load the trained model and scaler
with open("churn_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]

# Function to make predictions
def predict_churn(features):
    # Scale the features using the loaded scaler
    scaled_features = scaler.transform([features])
    return model.predict(scaled_features)[0]

# Streamlit interface configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔍",
    layout="centered",
)

# Header section
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1rem;
        }
        .subtitle {
            font-size: 1.2rem;
            text-align: center;
            color: gray;
            margin-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict the likelihood of a customer leaving your service</div>', unsafe_allow_html=True)

# Add an attractive banner image
image = Image.open("Blog-12.jpg")
st.image(image, use_container_width=True)

# Input fields section
st.sidebar.header("🔍 Input Customer Details")
st.sidebar.markdown("Fill out the details below to get a prediction:")

# Sidebar inputs
customer_id = st.sidebar.text_input("Customer ID")
firstname = st.sidebar.text_input("First Name")
lastname = st.sidebar.text_input("Last Name")
credit_score = st.sidebar.number_input("Credit Score", min_value=0, max_value=1000, step=1)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1)
tenure = st.sidebar.number_input("Tenure", min_value=0, max_value=100, step=1)
balance = st.sidebar.number_input("Balance", min_value=0.0, step=100.0)
num_of_products = st.sidebar.number_input("Number of Products", min_value=1, max_value=20, step=1)
has_cr_card = st.sidebar.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.sidebar.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, step=1000.0)

# Convert categorical fields to numerical values
def process_input_data():
    gender_map = {"Male": 1, "Female": 0}
    yes_no_map = {"Yes": 1, "No": 0}

    # OneHotEncoding for Geography
    geo_fr = {"France": 1, "Germany": 0, "Spain": 0}
    geo_ger = {"France": 0, "Germany": 1, "Spain": 0}
    geo_sp = {"France": 0, "Germany": 0, "Spain": 1}
    france = geo_fr[geography]
    ger = geo_ger[geography]
    spain = geo_sp[geography]
    Balance_Salary_Ratio = balance / estimated_salary

    return [
        credit_score,
        gender_map[gender],
        age,
        tenure,
        num_of_products,
        yes_no_map[has_cr_card],
        yes_no_map[is_active_member],
        france,
        ger,
        spain,
        Balance_Salary_Ratio,
    ]

# Prediction button
st.markdown("### 🔍 Predict Customer Churn")
if st.button("Get Prediction"):
    features = process_input_data()
    
    # Predict churn using the scaled features
    prediction = predict_churn(features)
    
    if prediction == 1:
        st.markdown(
            """
            <div style="background-color:#FFCCCB;padding:15px;border-radius:10px;">
                <h3 style="color:#B22222;text-align:center;">❌ The customer is forecasted to stop using the service.</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color:#DFF2BF;padding:15px;border-radius:10px;">
                <h3 style="color:#4F8A10;text-align:center;">✅ The customer is forecasted to remain active.</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Footer section
st.markdown(
    """
    ---
    #### About the Model
    This application uses a state-of-the-art machine learning model trained on customer data to predict the likelihood of churn. 
    """
)
