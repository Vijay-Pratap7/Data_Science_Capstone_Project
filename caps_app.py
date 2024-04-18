import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the data
df = pd.read_csv("CAR DETAILS.csv")

# Load the trained model
with open('rfmodel.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Function to predict car price
def predict_price(car_name, year, km_driven, fuel, seller_type, transmission, owner):
    # Filter data for the selected car name
    car_data = df[df['name'] == car_name]

    # Prepare input features
    car_age = 2023 - year  # Calculate car age
    km_driven = km_driven
    fuel_diesel = 1 if fuel == 'Diesel' else 0
    fuel_petrol = 1 if fuel == 'Petrol' else 0
    seller_type_individual = 1 if seller_type == 'Individual' else 0
    seller_type_dealer = 1 if seller_type == 'Dealer' else 0
    transmission_manual = 1 if transmission == 'Manual' else 0
    owner_first = 1 if owner == 'First Owner' else 0
    owner_second = 1 if owner == 'Second Owner' else 0

    # Make prediction
    input_data = np.array([[car_age, km_driven, fuel_diesel, fuel_petrol, seller_type_individual,
                            seller_type_dealer, transmission_manual, owner_first, owner_second]])

    prediction = rf_model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title("Car Price Prediction")

    # Select car name from dropdown list
    car_name = st.selectbox("Select Car Name", df['name'].unique())

    # Get car details based on selected car name
    selected_car = df[df['name'] == car_name].iloc[0]

    # Input features
    year = st.number_input("Manufacturing Year", min_value=1980, max_value=2022, value=selected_car['year'])
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=selected_car['km_driven'])
    fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel'], index=0 if selected_car['fuel'] == 'Petrol' else 1)
    seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer'], index=0 if selected_car['seller_type'] == 'Individual' else 1)
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'], index=0 if selected_car['transmission'] == 'Manual' else 1)
    owner = st.selectbox("Owner", ['First Owner', 'Second Owner'], index=0 if selected_car['owner'] == 'First Owner' else 1)

    # Predict price
    if st.button("Predict"):
        price = predict_price(car_name, year, km_driven, fuel, seller_type, transmission, owner)
        st.success(f"The estimated selling price of the car is ₹ {price:,.2f}")

if __name__ == '__main__':
    main()
