import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load dataset for feature ranges
data = pd.read_csv("C:\\Users\\aayus\\ml project house prediction\\data.csv")

# Load model
model = joblib.load("C:\\Users\\aayus\\ml project house prediction\\Real .joblib")

# Define features (assumes last column is the target)
features = data.columns[:-1]  # all except 'MEDV'

st.set_page_config(page_title="House Price Prediction", layout="centered")
st.title("🏠 House Price Predictor")

# Generate random input based on feature min-max
input_data = []
for feature in features:
    min_val = data[feature].min()
    max_val = data[feature].max()
    if data[feature].dtype == 'int64' or data[feature].nunique() < 10:
        value = np.random.randint(min_val, max_val + 1)
    else:
        value = np.random.uniform(min_val, max_val)
    input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)
input_df = pd.DataFrame(input_array, columns=features)

# Show generated input
st.subheader("🔢 Randomly Generated Inputs")
st.dataframe(input_df)

# Predict price
prediction = model.predict(input_array)[0]
st.subheader("💰 Predicted House Price")
st.success(f"${prediction * 1000:,.2f}")

# Option to rerun
if st.button("🔁 Generate New Prediction"):
    st.rerun()  # Updated to st.rerun() instead of experimental_rerun()
