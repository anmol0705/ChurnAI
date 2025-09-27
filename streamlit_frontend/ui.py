import streamlit as st
import requests
from PIL import Image

# --- 1. Streamlit Interface Configuration ---
st.set_page_config(
    page_title="ChurnAI",
    page_icon="🔍",
    layout="centered",
)

# --- 2. Custom CSS and Header ---
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem; font-weight: 700; text-align: center; margin-bottom: 1rem;
        }
        .subtitle {
            font-size: 1.2rem; text-align: center; color: gray; margin-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="main-title">Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict the likelihood of a customer leaving your service</div>', unsafe_allow_html=True)

# --- 3. Banner Image ---
# This will gracefully handle the case where the image is not found.
try:
    image = Image.open("Blog-12.jpg")
    st.image(image)
except FileNotFoundError:
    st.warning("Banner image 'Blog-12.jpg' not found. Please place it in the 'streamlit_frontend' folder.")

# --- 4. Sidebar Input Fields ---
st.sidebar.header("🔍 Input Customer Details")
st.sidebar.markdown("Fill out the details below to get a prediction:")

with st.sidebar:
    # Non-model inputs for UI context
    st.text_input("Customer ID")
    st.text_input("First Name")
    st.text_input("Last Name")

    # Model inputs
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=42)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
    balance = st.number_input("Balance", min_value=0.0, value=125000.0)
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0)

# --- 5. Prediction Button and API Call ---
st.markdown("### 🔮 Get Your Prediction")
if st.button("Predict Churn", type="primary"):
    # Create the JSON payload from the user's inputs
    payload = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0, # Convert to 0/1 for the API
        "IsActiveMember": 1 if is_active_member == "Yes" else 0, # Convert to 0/1
        "EstimatedSalary": estimated_salary,
        "Geography": geography,
        "Gender": gender
    }

    try:
        # Send the request to the FastAPI backend
        response = requests.post("https://anmol752005-churnai.hf.space/predict", json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        result = response.json()
        prediction = result['churn_prediction']

        # Display the result using the custom HTML boxes from your old script
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
                    <h3 style="color:#4F8A10;text-align:center;">✅  The customer is forecasted to remain active.</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except requests.exceptions.RequestException as e:
        st.error(f"Connection to the API failed. Please ensure the backend is running. Error: {e}")

# --- 6. Footer Section ---
st.markdown(
    """
    ---
    #### About the ChurnAI Model
    This application uses a machine learning service to predict customer churn based on their banking behavior and demographics.
    """
)
