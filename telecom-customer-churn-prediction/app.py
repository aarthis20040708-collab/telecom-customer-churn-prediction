import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load production assets
@st.cache_resource
def load_production_assets():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return model, scaler, encoders, feature_cols

model, scaler, encoders, feature_cols = load_production_assets()

st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")
st.title("📞 Telecom Customer Retention & Churn Risk Hub")
st.markdown("Adjust the real-time operational usage metrics below to predict customer cancellation risk profiles.")

st.divider()

# Organize dashboard into columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Account Profile")
    state = st.selectbox("Customer State", options=sorted(encoders['State'].classes_))
    account_length = st.number_input("Account Length (Weeks Active)", min_value=1, max_value=300, value=100)
    area_code = st.selectbox("Area Code", options=[408, 415, 510])
    intl_plan = st.radio("International Plan?", options=["No", "Yes"])
    vmail_plan = st.radio("Voice Mail Plan?", options=["No", "Yes"])
    vmail_msg = st.number_input("Number of Voicemail Messages", min_value=0, max_value=60, value=0)

with col2:
    st.subheader("☀️ Day & Evening Use")
    day_mins = st.number_input("Total Day Minutes", min_value=0.0, max_value=400.0, value=180.0)
    day_calls = st.number_input("Total Day Calls", min_value=0, max_value=200, value=100)
    # Estimate standard daytime rate (~$0.17 per minute)
    day_charge = day_mins * 0.17004
    st.caption(f"Calculated Day Charge: ${day_charge:.2f}")
    
    eve_mins = st.number_input("Total Evening Minutes", min_value=0.0, max_value=400.0, value=200.0)
    eve_calls = st.number_input("Total Evening Calls", min_value=0, max_value=200, value=100)
    eve_charge = eve_mins * 0.08501
    st.caption(f"Calculated Evening Charge: ${eve_charge:.2f}")

with col3:
    st.subheader("🌙 Night & International")
    night_mins = st.number_input("Total Night Minutes", min_value=0.0, max_value=400.0, value=200.0)
    night_calls = st.number_input("Total Night Calls", min_value=0, max_value=200, value=100)
    night_charge = night_mins * 0.045
    st.caption(f"Calculated Night Charge: ${night_charge:.2f}")
    
    intl_mins = st.number_input("Total International Minutes", min_value=0.0, max_value=50.0, value=10.0)
    intl_calls = st.number_input("Total International Calls", min_value=0, max_value=30, value=3)
    intl_charge = intl_mins * 0.27
    st.caption(f"Calculated Intl Charge: ${intl_charge:.2f}")
    
    st.divider()
    cust_service_calls = st.slider("Customer Service Calls Made", min_value=0, max_value=10, value=1)

# Compile raw input into dictionary
raw_input = {
    'State': encoders['State'].transform([state])[0],
    'Account length': account_length,
    'Area code': area_code,
    'International plan': encoders['International plan'].transform([intl_plan])[0],
    'Voice mail plan': encoders['Voice mail plan'].transform([vmail_plan])[0],
    'Number vmail messages': vmail_msg,
    'Total day minutes': day_mins,
    'Total day calls': day_calls,
    'Total day charge': day_charge,
    'Total eve minutes': eve_mins,
    'Total eve calls': eve_calls,
    'Total eve charge': eve_charge,
    'Total night minutes': night_mins,
    'Total night calls': night_calls,
    'Total night charge': night_charge,
    'Total intl minutes': intl_mins,
    'Total intl calls': intl_calls,
    'Total intl charge': intl_charge,
    'Customer service calls': cust_service_calls
}

# Turn into matching feature row
input_df = pd.DataFrame([raw_input])[feature_cols]

st.divider()

# Action Button
if st.button("Evaluate Churn Risk Profile", type="primary"):
    # Apply identical scaling rules used during training
    scaled_features = scaler.transform(input_df)
    
    # Calculate probability score
    prob = model.predict_proba(scaled_features)[0][1]
    risk_score = prob * 100
    
    st.header("🔍 Diagnostic Evaluation")
    
    if risk_score >= 70:
        st.error(f"🔴 CRITICAL RISK PROFILE: {risk_score:.1f}% Risk of Imminent Churn.")
        st.write("**Recommended Intervention:** Provide direct retention discounts. Flag account for account management outreach.")
    elif risk_score >= 35:
        st.warning(f"🟡 ELEVATED RISK PROFILE: {risk_score:.1f}% Risk of Churn.")
        st.write("**Recommended Intervention:** Trigger an automated engagement email or check-in questionnaire.")
    else:
        st.success(f"🟢 HEALTHY ACCOUNT PROFILE: {risk_score:.1f}% Churn Probability.")
        st.write("**Recommended Intervention:** Account remains stable. Excellent candidate for up-selling standard features.")