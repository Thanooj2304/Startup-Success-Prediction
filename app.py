import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Startup Success Predictor")
st.write("Enter the startup's financial details to predict its likelihood of success.")

# Input fields for the user
funding_total = st.number_input("Total Funding Raised (USD)", min_value=0, value=1000000)
funding_rounds = st.number_input("Number of Funding Rounds", min_value=0, value=2)
days_to_first_funding = st.number_input("Days from Founding to First Funding", min_value=0, value=180)
funding_duration = st.number_input("Funding Duration (Days between First and Last Round)", min_value=0, value=365)
founding_year = st.number_input("Year Founded", min_value=1980, max_value=2014, value=2010)
avg_funding_per_round = funding_total / (funding_rounds + 1)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([{
        'funding_total_usd': funding_total,
        'funding_rounds': funding_rounds,
        'days_to_first_funding': days_to_first_funding,
        'funding_duration_days': funding_duration,
        'founding_year': founding_year,
        'avg_funding_per_round': avg_funding_per_round
    }])

    input_scaled = scaler.transform(input_data)
    probability = model.predict_proba(input_scaled)[0][1]
    score = round(probability * 100, 1)

    st.subheader(f"Success Probability: {score}%")

    if score >= 75:
        st.success("✅ Tier 1 — Strong Buy: Fast-track to partner review")
    elif score >= 55:
        st.info("🔵 Tier 2 — Consider: Standard due diligence queue")
    elif score >= 40:
        st.warning("🟡 Tier 3 — Watch: Monitor for one more funding round")
    else:
        st.error("🔴 Tier 4 — Avoid: Decline without further review")
