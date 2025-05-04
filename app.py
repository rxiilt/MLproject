import streamlit as st
import joblib
import numpy as np
import lightgbm as lgb

# 1. Load the model
model = joblib.load('model.pkl')

st.title("Fraud Detection App")

# 2. Numeric inputs
amount            = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
quantity          = st.number_input("Quantity", min_value=1, step=1)
customer_age      = st.number_input("Customer Age", min_value=0, step=1)
account_age_days  = st.number_input("Account Age (Days)", min_value=0, step=1)
transaction_hour  = st.number_input("Transaction Hour (0–23)", min_value=0, max_value=23, step=1)

# 3. Categorical inputs — choices must match your training encodings
payment_methods     = ["credit_card", "paypal", "bank_transfer"]
product_categories  = ["electronics", "clothing", "groceries"]
devices             = ["mobile", "desktop", "tablet"]

pm_choice  = st.selectbox("Payment Method", payment_methods)
pc_choice  = st.selectbox("Product Category", product_categories)
dev_choice = st.selectbox("Device Used", devices)

# 4. Encode categories to numeric IDs exactly as during training
#    (replace these dicts with the mappings you used!)
pm_map  = {"credit_card": 0, "paypal": 1, "bank_transfer": 2}
pc_map  = {"electronics": 0,  "clothing": 1, "groceries": 2}
dev_map = {"mobile": 0,       "desktop": 1,  "tablet": 2}

pm_num  = pm_map[pm_choice]
pc_num  = pc_map[pc_choice]
dev_num = dev_map[dev_choice]

# 5. Assemble the feature vector in the correct order
features = np.array([[
    amount,         # 0
    quantity,       # 1
    customer_age,   # 2
    account_age_days,# 3
    transaction_hour,# 4
    pm_num,         # 5
    pc_num,         # 6
    dev_num         # 7
]])

# 6. Predict
if st.button("Predict"):
    pred = model.predict(features)[0]
    label = "Fraudulent" if pred == 1 else "Legitimate"
    st.success(f"Prediction: {label}")