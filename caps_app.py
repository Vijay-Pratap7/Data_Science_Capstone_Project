import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open('rfmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to predict car price
def predict_price(features):
    # Preprocess the input features (assuming they are in the same format as the original dataframe)
    df = pd.DataFrame(features, index=[0])
    # Make prediction
    prediction = model.predict(df)
    return prediction

# Streamlit UI
st.title("Car Price Prediction")

# Sidebar inputs
st.sidebar.header("Enter Car Details")
year = st.sidebar.number_input("Year", 1900, 2022, step=1)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Diesel", "Petrol", "CNG", "LPG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Individual", "Dealer"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])

# Transform sidebar inputs into features
features = {
    'year': year,
    'km_driven': km_driven,
    'fuel_type': fuel_type,
    'seller_type': seller_type,
    'transmission': transmission,
    'owner': owner
}

# Predict price on button click
if st.sidebar.button("Predict"):
    # Make prediction
    prediction = predict_price(features)
    st.success(f"The predicted car price is ₹ {prediction[0]:,.2f}")
