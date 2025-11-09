import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# App title
st.title("📦 Shipment Delivery Prediction App")

# User inputs
st.header("Enter Shipment Details")

warehouse_block = st.selectbox("Warehouse Block", ["A", "B", "C", "D", "F"])
mode_of_shipment = st.selectbox("Mode of Shipment", ["Flight", "Road", "Ship"])
customer_care_calls = st.number_input("Customer Care Calls", min_value=0, max_value=10, value=2)
customer_rating = st.slider("Customer Rating", 1, 5, 3)
cost_of_product = st.number_input("Cost of Product (in ₹)", min_value=1, value=200)
prior_purchases = st.number_input("Prior Purchases", min_value=0, value=1)
product_importance = st.selectbox("Product Importance", ["Low", "Medium", "High"])
gender = st.selectbox("Customer Gender", ["Male", "Female"])
discount_offered = st.number_input("Discount Offered (%)", min_value=0, max_value=100, value=10)
weight_in_gms = st.number_input("Weight (in grams)", min_value=1, value=500)

# Prediction button
if st.button("Predict"):
    # Compute cost-to-weight ratio
    cost_to_weight_ratio = cost_of_product / (weight_in_gms + 1e-6)

    # One-hot encode warehouse block (B, C, D, F)
    warehouse_block_B = 1 if warehouse_block == "B" else 0
    warehouse_block_C = 1 if warehouse_block == "C" else 0
    warehouse_block_D = 1 if warehouse_block == "D" else 0
    warehouse_block_F = 1 if warehouse_block == "F" else 0

    # One-hot encode mode of shipment (Road, Ship)
    mode_of_shipment_Road = 1 if mode_of_shipment == "Road" else 0
    mode_of_shipment_Ship = 1 if mode_of_shipment == "Ship" else 0

    # Encode other categorical values
    importance_map = {"Low": 0, "Medium": 1, "High": 2}
    gender_map = {"Male": 0, "Female": 1}

    importance_encoded = importance_map[product_importance]
    gender_encoded = gender_map[gender]

    # Construct input DataFrame with correct column names
    input_data = pd.DataFrame([[ 
        0,  # ID (dummy placeholder)
        customer_care_calls,
        customer_rating,
        cost_of_product,
        prior_purchases,
        importance_encoded,
        gender_encoded,
        discount_offered,
        weight_in_gms,
        warehouse_block_B,
        warehouse_block_C,
        warehouse_block_D,
        warehouse_block_F,
        mode_of_shipment_Road,
        mode_of_shipment_Ship,
        cost_to_weight_ratio
    ]], columns=[
        "ID",
        "Customer_care_calls",
        "Customer_rating",
        "Cost_of_the_Product",
        "Prior_purchases",
        "Product_importance",
        "Gender",
        "Discount_offered",
        "Weight_in_gms",
        "Warehouse_block_B",
        "Warehouse_block_C",
        "Warehouse_block_D",
        "Warehouse_block_F",
        "Mode_of_Shipment_Road",
        "Mode_of_Shipment_Ship",
        "Cost_to_Weight_ratio"
    ])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Display results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("✅ Shipment will reach on time")
    else:
        st.error("❌ Shipment will NOT reach on time")

    st.subheader("Prediction Probability")
    st.write(f"On-time probability: {probability[1]*100:.2f}%")
    st.write(f"Delay probability: {probability[0]*100:.2f}%")
