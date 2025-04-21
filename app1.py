import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# === Page Setup ===
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 Boston House Price Predictor")

# === Load Dataset ===
DATA_PATH = "C:\\Users\\aayus\\ml project house prediction\\data.csv"
if not os.path.exists(DATA_PATH):
    st.error("Dataset not found!")
    st.stop()

data = pd.read_csv(DATA_PATH)
features = data.columns[:-1]  # all except MEDV

# === Load Model ===
MODEL_PATH = "C:\\Users\\aayus\\ml project house prediction\\Real .joblib"
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found!")
    st.stop()

model = joblib.load(MODEL_PATH)

# === Generate Random Input from Realistic Feature Ranges ===
input_data = []
for feature in features:
    min_val = data[feature].min()
    max_val = data[feature].max()
    if data[feature].dtype == 'int64' or data[feature].nunique() <= 10:
        value = np.random.randint(min_val, max_val + 1)
    else:
        value = np.random.uniform(min_val, max_val)
    input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)
input_df = pd.DataFrame(input_array, columns=features)

# === Show Input Features ===
st.subheader("🔢 Randomly Generated Features")
st.dataframe(input_df)

# === Predict Price ===
try:
    prediction = model.predict(input_array)[0]
    price = max(prediction, 0)  # Ensure non-negative
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# === Show Prediction ===
st.subheader("💰 Predicted House Price")
st.success(f"Estimated Price: **${price * 1000:,.2f}**")

# === Regenerate Button ===
if st.button("🔁 Generate New Prediction"):
    st.query_params["refresh"] = str(np.random.rand())  # Refresh with random query param
