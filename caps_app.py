import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open('rfmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataframe
df = pd.read_csv("CAR DETAILS.csv")

# Function to preprocess features
def preprocess_features(features):
    df_features = pd.DataFrame(features, index=[0])
    df_features["car_age"] = 2023 - df_features["year"]
    name = df_features["name"].str.split(" ", expand=True)
    df_features["car_maker"] = name[0]
    df_features["car_model"] = name[1]
    df_features.drop(["name"], axis=1, inplace=True)
    df_features = pd.get_dummies(df_features, drop_first=True)
    return df_features

# Function to predict car price
def predict_price(features):
    df_features = preprocess_features(features)
    prediction = model.predict(df_features)
    return prediction

# Streamlit UI
st.title("Car Price Prediction")

# Sidebar inputs
st.sidebar.header("Enter Car Details")
if "name" in df.columns:  # Check if 'name' column exists in the dataframe
    car_name = st.sidebar.selectbox("Car Name", df["name"].unique())
else:
    st.sidebar.error("No 'name' column found in the dataframe. Please check your data source.")
year = st.sidebar.number_input("Year", 1900, 2022, step=1)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0)
fuel_type = st.sidebar.selectbox("Fuel Type", df["fuel"].unique() if "fuel" in df.columns else [])
seller_type = st.sidebar.selectbox("Seller Type", df["seller_type"].unique() if "seller_type" in df.columns else [])
transmission = st.sidebar.selectbox("Transmission", df["transmission"].unique() if "transmission" in df.columns else [])
owner = st.sidebar.selectbox("Owner", df["owner"].unique() if "owner" in df.columns else [])

# Transform sidebar inputs into features
features = {
    'year': year,
    'km_driven': km_driven,
    'fuel_type': fuel_type,
    'seller_type': seller_type,
    'transmission': transmission,
    'owner': owner
}

# Predict price on button click
if st.sidebar.button("Predict"):
    # Make prediction
    prediction = predict_price(features)
    st.success(f"The predicted car price is ₹ {prediction[0]:,.2f}")
