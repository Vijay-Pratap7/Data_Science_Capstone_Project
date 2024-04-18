import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('rfmodel.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Function to predict car price
def predict_price(car_age, km_driven, fuel, seller_type, transmission, owner, car_maker):
    # Prepare input features
    fuel_diesel = 1 if fuel == 'Diesel' else 0
    fuel_petrol = 1 if fuel == 'Petrol' else 0
    seller_type_individual = 1 if seller_type == 'Individual' else 0
    seller_type_dealer = 1 if seller_type == 'Dealer' else 0
    transmission_manual = 1 if transmission == 'Manual' else 0
    owner_first = 1 if owner == 'First Owner' else 0
    owner_second = 1 if owner == 'Second Owner' else 0
    car_maker_encoded = [0, 0, 0, 0]  # Assuming 4 car makers for one-hot encoding
    if car_maker in ['Maruti', 'Hyundai', 'Datsun', 'Honda']:
        car_maker_encoded = [1 if maker == car_maker else 0 for maker in ['Maruti', 'Hyundai', 'Datsun', 'Honda']]

    # Make prediction
    input_data = np.array([[car_age, km_driven, fuel_diesel, fuel_petrol, seller_type_individual,
                            seller_type_dealer, transmission_manual, owner_first, owner_second] + car_maker_encoded])
    
    prediction = rf_model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title("Car Price Prediction")

    # Input features
    car_age = st.slider("Car Age", 0, 20, 5)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=10000)
    fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel'])
    seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    owner = st.selectbox("Owner", ['First Owner', 'Second Owner'])
    car_maker = st.selectbox("Car Maker", ['Maruti', 'Hyundai', 'Datsun', 'Honda'])

    # Predict price
    if st.button("Predict"):
        price = predict_price(car_age, km_driven, fuel, seller_type, transmission, owner, car_maker)
        st.success(f"The estimated selling price of the car is ₹ {price:,.2f}")

if __name__ == '__main__':
    main()
