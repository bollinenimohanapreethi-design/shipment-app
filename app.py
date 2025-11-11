import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

st.title("Shipment On-Time Prediction")
st.write("Predict whether a shipment will reach on time using shipment and customer features.")

st.sidebar.header("Enter Input Features")

# Input features (user-entered)
Customer_care_calls = st.sidebar.number_input("Customer Care Calls", min_value=1, step=1)
Customer_rating = st.sidebar.slider("Customer Rating", min_value=1, max_value=5, value=3)
Cost_of_the_Product = st.sidebar.number_input("Cost of the Product", min_value=1)
Prior_purchases = st.sidebar.number_input("Prior Purchases", min_value=0, step=1)
Product_importance = st.sidebar.selectbox("Product Importance", ['low', 'medium', 'high'])
Gender = st.sidebar.selectbox("Gender", ['Female', 'Male'])
Discount_offered = st.sidebar.number_input("Discount Offered", min_value=0)
Weight_in_gms = st.sidebar.number_input("Weight in grams", min_value=1)
Cost_to_Weight_ratio = st.sidebar.number_input("Cost-to-Weight Ratio", min_value=0.0, format="%.4f")

# One-hot encoding for Warehouse_block (B, C, D, F) – drop A as base case
warehouse_options = ['A', 'B', 'C', 'D', 'F']
selected_warehouse = st.sidebar.selectbox("Warehouse Block", warehouse_options)
Warehouse_block_B = 1 if selected_warehouse == 'B' else 0
Warehouse_block_C = 1 if selected_warehouse == 'C' else 0
Warehouse_block_D = 1 if selected_warehouse == 'D' else 0
Warehouse_block_F = 1 if selected_warehouse == 'F' else 0

# One-hot encoding for Mode_of_Shipment (Road, Ship) – drop Flight as base case
mode_options = ['Flight', 'Road', 'Ship']
selected_mode = st.sidebar.selectbox("Mode of Shipment", mode_options)
Mode_of_Shipment_Road = 1 if selected_mode == 'Road' else 0
Mode_of_Shipment_Ship = 1 if selected_mode == 'Ship' else 0

# Encode Product_importance and Gender as during training (assuming label encoding: low=0, medium=1, high=2; Female=0, Male=1)
importance_dict = {'low': 0, 'medium': 1, 'high': 2}
gender_dict = {'Female': 0, 'Male': 1}

# Assemble feature vector in the correct order
features = [
    Customer_care_calls,
    Customer_rating,
    Cost_of_the_Product,
    Prior_purchases,
    importance_dict[Product_importance],
    gender_dict[Gender],
    Discount_offered,
    Weight_in_gms,
    # 'Reached.on.Time_Y.N' is your target, not part of the input for prediction
    Warehouse_block_B,
    Warehouse_block_C,
    Warehouse_block_D,
    Warehouse_block_F,
    Mode_of_Shipment_Road,
    Mode_of_Shipment_Ship,
    Cost_to_Weight_ratio
]

if st.button("Predict"):
    X = np.array([features])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    st.success(f"Prediction: {'On Time' if prediction == 1 else 'Not On Time'}")
    st.info(f"Probability of on-time delivery: {probability:.2f}")

st.markdown("---")
st.markdown("**Caution:** Ensure all encodings follow your training preprocessing for consistent results.")
