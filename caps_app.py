import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('rfmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the encoder
with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Function to predict car price
def predict_price(year, km_driven, fuel, seller_type, transmission, owner):
    fuel_encoded = encoder.transform([fuel])[0]
    seller_type_encoded = encoder.transform([seller_type])[0]
    transmission_encoded = encoder.transform([transmission])[0]
    owner_encoded = encoder.transform([owner])[0]

    car_details = np.array([year, km_driven, fuel_encoded, seller_type_encoded, transmission_encoded, owner_encoded]).reshape(1, -1)
    prediction = model.predict(car_details)
    return prediction[0]

# Streamlit UI
def main():
    st.title('Car Price Prediction')
    st.write('This app predicts the price of a car based on its details.')

    year = st.slider('Year', 1990, 2023, 2010)
    km_driven = st.number_input('Kilometers Driven', min_value=0)
    fuel = st.selectbox('Fuel Type', ['Diesel', 'Petrol', 'CNG', 'LPG'])
    seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

    if st.button('Predict'):
        prediction = predict_price(year, km_driven, fuel, seller_type, transmission, owner)
        st.success(f'Predicted Price: ₹{prediction:,.2f}')

if __name__ == "__main__":
    main()
