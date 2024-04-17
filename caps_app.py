import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('rfmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to predict car price
def predict_price(year, km_driven, fuel, seller_type, transmission, owner):
    # Encode categorical variables
    fuel_encoded = 1 if fuel == 'Petrol' else 0  # Assuming petrol is encoded as 1
    seller_type_encoded = 1 if seller_type == 'Dealer' else 0  # Assuming dealer is encoded as 1
    transmission_encoded = 1 if transmission == 'Automatic' else 0  # Assuming automatic is encoded as 1
    owner_encoded = 0  # No need to encode for owner as it's already numeric

    car_details = np.array([year, km_driven, fuel_encoded, seller_type_encoded, transmission_encoded, owner_encoded]).reshape(1, -1)
    prediction = model.predict(car_details)
    return prediction[0]

# Streamlit UI
def main():
    st.title('Car Price Prediction')
    st.write('This app predicts the price of a car based on its details.')

    year = st.slider('Year', 1990, 2023, 2010)
    km_driven = st.number_input('Kilometers Driven', min_value=0)
    fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG'])
    seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

    if st.button('Predict'):
        prediction = predict_price(year, km_driven, fuel, seller_type, transmission, owner)
        st.success(f'Predicted Price: ₹{prediction:,.2f}')

if __name__ == "__main__":
    main()
