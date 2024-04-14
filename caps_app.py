import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the pre-trained Random Forest model
with open('rfmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the DataFrame with car details
df = pd.read_csv("CAR DETAILS.csv")

# Relevant features used during training
relevant_features = ['km_driven', 'year', 'fuel_Diesel', 'fuel_LPG', 'seller_type_Individual', 'seller_type_Trustmark_Dealer', 'transmission_Manual', 'owner_Fourth_and_Above_Owner', 'owner_Second_Owner', 'owner_Test_Drive_Car', 'owner_Third_Owner']

# Label encoding for categorical variables
label_encoders = {}
for col in ['fuel', 'seller_type', 'transmission', 'owner']:
    label_encoders[col] = {val: idx for idx, val in enumerate(df[col].unique())}

# Streamlit app
def main():
    st.title('Used Car Price Prediction')

    # Create a selectbox for car names
    car_name = st.selectbox('Car Name', df['name'])

    # Get car details based on the selected car name
    car_details = df[df['name'] == car_name].iloc[0]

    # Input fields for user to enter other car details
    km_driven = st.number_input('Kilometers Driven', value=50000)
    year = st.number_input('Year of Purchase', min_value=1990, max_value=2023, value=2015)
    fuel = st.selectbox('Fuel Type', list(label_encoders['fuel'].keys()))
    seller_type = st.selectbox('Seller Type', list(label_encoders['seller_type'].keys()))
    transmission = st.selectbox('Transmission', list(label_encoders['transmission'].keys()))
    owner = st.selectbox('Owner', list(label_encoders['owner'].keys()))

    # Function to predict the price based on user input
    def predict_price(km_driven, year, fuel, seller_type, transmission, owner):
        input_data = np.array([km_driven, year, label_encoders['fuel'][fuel], label_encoders['seller_type'][seller_type], label_encoders['transmission'][transmission], label_encoders['owner'][owner]])
        predicted_price = model.predict([input_data])
        return predicted_price

    # Predict the price when the user clicks the 'Predict' button
    if st.button('Predict'):
        predicted_price = predict_price(km_driven, year, fuel, seller_type, transmission, owner)
        st.success(f'Predicted Price: {predicted_price[0]:,.2f} INR')

if __name__ == '__main__':
    main()
