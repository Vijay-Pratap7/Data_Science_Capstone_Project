import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('rfmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to predict car price
def predict_price(car_details):
    car_details_encoded = model.transform(car_details)
    prediction = model.predict(car_details_encoded.reshape(1, -1))
    return prediction[0]

# Streamlit UI
def main():
    st.title('Car Price Prediction')

    st.write('This app predicts the price of a car based on its details.')

    # Input form
    st.sidebar.header('Enter Car Details:')
    year = st.sidebar.slider('Year', 1990, 2023, 2010)
    km_driven = st.sidebar.number_input('Kilometers Driven', min_value=0)
    fuel = st.sidebar.selectbox('Fuel Type', ['Diesel', 'Petrol', 'CNG', 'LPG'])
    seller_type = st.sidebar.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.sidebar.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

    # Encode categorical variables
    fuel_encoded = encoder.transform([fuel])[0]
    seller_type_encoded = encoder.transform([seller_type])[0]
    transmission_encoded = encoder.transform([transmission])[0]
    owner_encoded = encoder.transform([owner])[0]

    # Prepare car details as a dictionary
    car_details = {
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel_encoded,
        'seller_type': seller_type_encoded,
        'transmission': transmission_encoded,
        'owner': owner_encoded
    }

    # Predict car price on button click
    if st.sidebar.button('Predict'):
        prediction = predict_price(car_details)
        st.sidebar.success(f'Predicted Price: ₹{prediction:,.2f}')

if __name__ == "__main__":
    main()
