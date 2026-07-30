import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
model = joblib.load(model_path)

# App title and description
st.title("Wellness Tourism Package Prediction")
st.write("Predict whether a customer will purchase the Wellness Tourism Package.")
st.write("---")

# Collect user inputs
st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
    number_of_followups = st.number_input("Number of Followups", min_value=1, max_value=10, value=3)
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])

with col2:
    number_of_trips = st.number_input("Number of Trips (Annual)", min_value=1, max_value=20, value=2)
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=5000, max_value=100000, value=20000)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)

st.write("---")

# Create input dataframe
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.write("---")
    if prediction == 1:
        st.success(f"The customer is LIKELY to purchase the package. (Probability: {probability[1]:.2%})")
    else:
        st.error(f"The customer is UNLIKELY to purchase the package. (Probability of purchase: {probability[1]:.2%})")

    st.write("**Input Summary:**")
    st.dataframe(input_data.T.rename(columns={0: "Value"}))
