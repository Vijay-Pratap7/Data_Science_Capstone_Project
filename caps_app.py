import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the pre-trained Random Forest model
with open('rfmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the DataFrame with car details
df = pd.read_csv("CAR DETAILS.csv")

# Streamlit app
def main():
    st.title('Used Car Price Prediction')

    # Create a selectbox for car names
    car_name = st.selectbox('Car Name', df['name'])

    # Get car details based on the selected car name
    car_details = df[df['name'] == car_name].iloc[0]

    # Input fields for user to enter other car details
    car_age = st.slider('Car Age', min_value=1, max_value=20, value=5)
    km_driven = st.number_input('Kilometers Driven', value=50000)
    year = st.number_input('Year of Purchase', min_value=1990, max_value=2023, value=2015)
    fuel_Diesel = st.checkbox('Fuel Type: Diesel')
    fuel_LPG = st.checkbox('Fuel Type: LPG')
    seller_type_Individual = st.checkbox('Seller Type: Individual')
    seller_type_Trustmark_Dealer = st.checkbox('Seller Type: Trustmark Dealer')
    transmission_Manual = st.checkbox('Transmission: Manual')
    owner_Fourth_and_Above_Owner = st.checkbox('Owner: Fourth and Above Owner')
    owner_Second_Owner = st.checkbox('Owner: Second Owner')
    owner_Test_Drive_Car = st.checkbox('Owner: Test Drive Car')
    owner_Third_Owner = st.checkbox('Owner: Third Owner')

    # Function to predict the price based on user input
    def predict_price(car_details, car_age, km_driven, year, fuel_Diesel, fuel_LPG, seller_type_Individual, seller_type_Trustmark_Dealer, transmission_Manual, owner_Fourth_and_Above_Owner, owner_Second_Owner, owner_Test_Drive_Car, owner_Third_Owner):
        input_data = np.array([car_age, km_driven, year, fuel_Diesel, fuel_LPG, seller_type_Individual, seller_type_Trustmark_Dealer, transmission_Manual, owner_Fourth_and_Above_Owner, owner_Second_Owner, owner_Test_Drive_Car, owner_Third_Owner]).reshape(1, -1)
        predicted_price = model.predict(input_data)
        return predicted_price

    # Predict the price when the user clicks the 'Predict' button
    if st.button('Predict'):
        predicted_price = predict_price(car_details, car_age, km_driven, year, fuel_Diesel, fuel_LPG, seller_type_Individual, seller_type_Trustmark_Dealer, transmission_Manual, owner_Fourth_and_Above_Owner, owner_Second_Owner, owner_Test_Drive_Car, owner_Third_Owner)
        st.success(f'Predicted Price: {predicted_price[0]:,.2f} INR')

if __name__ == '__main__':
    main()
