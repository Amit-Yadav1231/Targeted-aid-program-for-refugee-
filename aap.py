import streamlit as st
import pandas as pd
import joblib
import numpy as np
from catboost import CatBoostClassifier

# Load model
model = CatBoostClassifier()
model.load_model("catboost_sector_model.cbm")

# Load encoder & feature order
encoder = joblib.load("ordinal_encoder.pkl")
features = joblib.load("model_features.pkl")

st.title("Humanitarian Sector Prediction System")

population_status = st.selectbox("Population Status", ["Adult", "Children"])
category = st.selectbox("Category", ["INN", "TGT"])
population = st.number_input("Population", min_value=0)
admin_level = st.selectbox("Admin Level", [0, 1, 2])
data_year = st.selectbox("Year", [2024, 2025])

if st.button("Predict Sector"):
    input_df = pd.DataFrame([{
        "population_status": population_status,
        "category": category,
        "population": population,
        "admin_level": admin_level,
        "data_year": data_year
    }])

    # Ensure same column order
    input_df = input_df[features]

    # Encode categorical features
    cat_cols = input_df.select_dtypes(include="object").columns
    input_df[cat_cols] = encoder.transform(input_df[cat_cols])

    # Top-3 prediction logic
    probs = model.predict_proba(input_df)[0]

    top_k = 3
    top_indices = np.argsort(probs)[::-1][:top_k]

    top_sectors = [model.classes_[i] for i in top_indices]
    top_scores = [probs[i] for i in top_indices]

    st.success("Recommended Humanitarian Sectors:")
    for sec, score in zip(top_sectors, top_scores):
        st.write(f"• {sec}  (confidence: {score:.2f})")

