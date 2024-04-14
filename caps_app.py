import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    # Feature engineering
    data["car_age"] = 2023 - data["Year"]
    name = data["name"].str.split(" ", expand=True)
    data["car_maker"] = name[0]
    data["car_model"] = name[1]
    data.drop(["name"], axis=1, inplace=True)
    
    # Encoding categorical variables
    data = pd.get_dummies(data, drop_first=True)
    return data

# Function to predict car price
def predict_price(car_data):
    car_data_processed = preprocess_input(car_data)
    prediction = model.predict(car_data_processed)
    return prediction

# Streamlit app
def main():
    st.title("Used Car Price Prediction")
    
    # Input form
    st.sidebar.header("Enter Car Details")
    
    year = st.sidebar.number_input("Year of Manufacture", min_value=1900, max_value=2023, step=1)
    km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, step=1)
    fuel = st.sidebar.selectbox("Fuel Type", ["Diesel", "Petrol", "CNG", "LPG"])
    seller_type = st.sidebar.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.sidebar.selectbox("Transmission Type", ["Manual", "Automatic"])
    owner = st.sidebar.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner or More"])
    
    car_data = pd.DataFrame({
        "year": [year],
        "km_driven": [km_driven],
        "fuel_" + fuel: [1],
        "seller_type_" + seller_type: [1],
        "transmission_" + transmission: [1],
        "owner_" + owner: [1]
    })
    
    if st.sidebar.button("Predict"):
        # Predict car price
        prediction = predict_price(car_data)
        st.sidebar.success(f"The estimated selling price of the car is ₹{prediction[0]}")

if __name__ == "__main__":
    main()
