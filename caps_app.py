import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the trained model
with open('best_model1.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
def main():
    st.title('Car Selling Price Prediction')

    # Upload CSV file
    st.sidebar.header('Upload Dataset')
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Dropdown list for car_maker
        car_maker_options = df['car_maker'].unique()
        car_maker = st.sidebar.selectbox('Car Maker', [''] + list(car_maker_options))

        # Dropdown list for car_model based on selected car_maker
        if car_maker != '':
            car_model_options = df[df['car_maker'] == car_maker]['car_model'].unique()
            car_model = st.sidebar.selectbox('Car Model', [''] + list(car_model_options))
        else:
            car_model = ''

        # Input form
        st.sidebar.header('Input Features')
        fuel = st.sidebar.selectbox('Fuel', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
        seller_type = st.sidebar.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
        transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
        owner = st.sidebar.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
        year = st.sidebar.number_input('Year', min_value=1980, max_value=2023)
        km_driven = st.sidebar.number_input('Kilometers Driven', min_value=0)

        # Prepare input data
        input_data = pd.DataFrame({
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner],
            'car_maker': [car_maker],
            'car_model': [car_model],
            'year': [year],
            'km_driven': [km_driven]
        })

        # Predict selling price
        if st.sidebar.button('Predict'):
            prediction = model.predict(input_data)
            st.sidebar.header('Prediction')
            st.sidebar.write(f'Predicted Selling Price: {prediction[0]}')

if __name__ == '__main__':
    main()
