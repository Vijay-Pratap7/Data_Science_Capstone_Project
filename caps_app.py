import streamlit as st
import pickle
import pandas as pd

# Load the saved trained ML model
with open('rfmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
      # Create a DataFrame from the input data
    input_df = pd.DataFrame(data)

    # Calculate car age
    input_df["car_age"] = 2023 - input_df["year"]

    # Extract car maker and model from 'name' column
    name = input_df["name"].str.split(" ", expand=True)
    input_df["car_maker"] = name[0]
    input_df["car_model"] = name[1]

    # Drop unnecessary columns
    input_df.drop(["name"], axis=1, inplace=True)

    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Encode categorical columns
    encoder = LabelEncoder()
    input_df = input_df.apply(encoder.fit_transform)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    return input_df

# Function to predict car price
def predict_price(car_data):
    # Preprocess input data
    car_data = preprocess_input(car_data)
    # Predict the price using the loaded model
    predicted_price = model.predict(car_data)
    return predicted_price

# Streamlit UI
st.title('Car Price Prediction')

# Input form for car details
st.header('Enter Car Details')
year = st.number_input('Year', min_value=1900, max_value=2023)
km_driven = st.number_input('Kilometers Driven')
fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner or More'])
car_maker = st.text_input('Car Maker')
car_model = st.text_input('Car Model')

# When predict button is clicked
if st.button('Predict'):
    # Create a dictionary with the input data
    input_data = {
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'car_maker': car_maker,
        'car_model': car_model
    }
    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_data])
    # Predict the price
    predicted_price = predict_price(input_df)
    # Display the predicted price
    st.success(f'Predicted Selling Price: Rs. {predicted_price[0]:,.2f}')
