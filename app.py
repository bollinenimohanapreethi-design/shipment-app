import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Title and description
st.title("Shipment On-Time Prediction")
st.write("""
This app predicts whether a shipment will reach on time or not.
Provide the shipment details in the sidebar and click predict!
""")

# Sidebar input section
st.sidebar.header("Enter Shipment Details")

# --- Categorical Inputs ---
warehouse_block = st.sidebar.selectbox("Warehouse Block", ["A", "B", "C", "D", "E"])
mode_of_shipment = st.sidebar.selectbox("Mode of Shipment", ["Ship", "Flight", "Road"])
product_importance = st.sidebar.selectbox("Product Importance", ["Low", "Medium", "High"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# --- Numerical Inputs ---
customer_care_calls = st.sidebar.number_input("Customer Care Calls", min_value=0, max_value=100, value=1)
customer_rating = st.sidebar.number_input("Customer Rating", min_value=1, max_value=5, value=3)
cost_of_product = st.sidebar.number_input("Cost of the Product", min_value=0, value=1000)
prior_purchases = st.sidebar.number_input("Prior Purchases", min_value=0, value=0)
discount_offered = st.sidebar.number_input("Discount Offered", min_value=0, max_value=100, value=10)
weight_in_gms = st.sidebar.number_input("Weight in grams", min_value=0, value=500)

# Map categorical features (example encoding)
warehouse_map = {"A":0, "B":1, "C":2, "D":3, "E":4}
mode_map = {"Ship":0, "Flight":1, "Road":2}
importance_map = {"Low":0, "Medium":1, "High":2}
gender_map = {"Male":0, "Female":1}

warehouse_encoded = warehouse_map[warehouse_block]
mode_encoded = mode_map[mode_of_shipment]
importance_encoded = importance_map[product_importance]
gender_encoded = gender_map[gender]

# Button for prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[warehouse_encoded, mode_encoded, customer_care_calls,
                                customer_rating, cost_of_product, prior_purchases,
                                importance_encoded, gender_encoded, discount_offered,
                                weight_in_gms]],
                              columns=["Warehouse_block", "Mode_of_Shipment", "Customer_care_calls",
                                       "Customer_rating", "Cost_of_the_Product", "Prior_purchases",
                                       "Product_importance", "Gender", "Discount_offered",
                                       "Weight_in_gms"])
    
    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.subheader("Prediction Result")
    st.write("✅ Shipment will reach on time" if prediction==1 else "❌ Shipment will NOT reach on time")
    
    st.subheader("Prediction Probability")
    st.write(f"On-time probability: {probability[1]*100:.2f}%")
    st.write(f"Delay probability: {probability[0]*100:.2f}%")
