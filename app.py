import streamlit as st
import joblib
import numpy as np
import lightgbm as lgb

# 1. Load the model
model = joblib.load('model.pkl')

# 2. App title
st.title("🔍 E-commerce Fraud Detection App")
st.markdown("Use this tool to predict whether a transaction is **fraudulent** or **legitimate** based on key features.")

# 3. Numeric inputs
amount = st.number_input("💵 Transaction Amount", min_value=0.0, format="%.2f")
quantity = st.number_input("📦 Quantity", min_value=1, step=1)
customer_age = st.number_input("👤 Customer Age", min_value=0, step=1)
account_age_days = st.number_input("📅 Account Age (in Days)", min_value=0, step=1)
transaction_hour = st.number_input("🕒 Transaction Hour (0–23)", min_value=0, max_value=23, step=1)

# 4. Categorical inputs
payment_methods = ["credit card", "debit card", "PayPal", "bank transfer"]
product_categories = ["electronics", "clothing", "home & garden", "toys & games", "health & beauty", "groceries"]
devices = ["mobile", "desktop", "tablet"]

pm_choice = st.selectbox("💳 Payment Method", payment_methods)
pc_choice = st.selectbox("🛍️ Product Category", product_categories)
dev_choice = st.selectbox("📱 Device Used", devices)

# 5. Category encoding (must match training-time encoding)
pm_map = {"credit card": 0, "debit card": 1, "PayPal": 2, "bank transfer": 3}
pc_map = {
    "electronics": 0, "clothing": 1, "home & garden": 2,
    "toys & games": 3, "health & beauty": 4, "groceries": 5
}
dev_map = {"mobile": 0, "desktop": 1, "tablet": 2}

pm_num = pm_map[pm_choice]
pc_num = pc_map[pc_choice]
dev_num = dev_map[dev_choice]

# 6. Feature vector
features = np.array([[amount, pm_num, pc_num, quantity, dev_num, customer_age, account_age_days, transaction_hour]])

# 7. Prediction
if st.button("🔎 Predict"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("🚨 Alert: Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
        st.balloons()
