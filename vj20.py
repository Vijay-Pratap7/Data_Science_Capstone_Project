import streamlit as st
import pandas as pd
import pickle

# Load the saved trained ML model
with open('rfmodel_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_price(features):
    # Create a DataFrame from the features dictionary
    input_data = pd.DataFrame([features])
    # Make predictions
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title('Car Selling Price Prediction')

    # Sidebar
    st.sidebar.title('Enter Car Details')
    year = st.sidebar.number_input('Year', min_value=1950, max_value=2023, step=1)
    km_driven = st.sidebar.number_input('Kilometers Driven', min_value=0)
    fuel_type = st.sidebar.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG'])
    seller_type = st.sidebar.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.sidebar.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

    # Make predictions based on user input
    features = {
        'year': year,
        'km_driven': km_driven,
        'fuel_type': fuel_type,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner
    }
    prediction = predict_price(features)

    st.write('## Predicted Selling Price:')
    st.write(f'Rs. {prediction[0]:,.2f}')

if __name__ == "__main__":
    main()
