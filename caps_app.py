import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data, X_train):
    # Feature engineering
    data["car_age"] = 2023 - data["year"]
    name = data["name"].str.split(" ", expand=True)
    data["car_maker"] = name[0]
    data["car_model"] = name[1]
    data.drop(["name"], axis=1, inplace=True)
    
    # Encoding categorical variables
    data = pd.get_dummies(data, drop_first=True)
    
    # Ensure all columns are present and in the same order as the training data
    missing_cols = set(X_train.columns) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
    
    # Reorder columns to match the order of training data
    data = data[X_train.columns]
    
    return data

# Function to predict car price
def predict_price(car_data):
    car_data_processed = preprocess_input(car_data, X_train)
    prediction = model.predict(car_data_processed)
    return prediction

# Streamlit app
def main():
    st.title("Used Car Price Prediction")
    
    # Read the data
    df = pd.read_csv("CAR DETAILS.csv")
    
    # Check if "selling_price" column exists in the dataset
    if "selling_price" not in df.columns:
        st.error("Error: 'selling_price' column not found in dataset.")
        return
    
    # Split the data into features and target
    X = df.drop(["selling_price"], axis=1)
    y = df["selling_price"]
    
    # Split the data into train and test sets
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Input form
    st.sidebar.header("Enter Car Details")
    
    year = st.sidebar.number_input("Year of Manufacture", min_value=1900, max_value=2023, step=1)
    km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, step=1)
    fuel = st.sidebar.selectbox("Fuel Type", ["Diesel", "Petrol", "CNG", "LPG"])
    seller_type = st.sidebar.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.sidebar.selectbox("Transmission Type", ["Manual", "Automatic"])
    owner = st.sidebar.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner or More"])
    car_names = df["name"].unique()
    selected_car_name = st.sidebar.selectbox("Car Name", car_names)
    
    car_data = pd.DataFrame({
        "year": [year],
        "km_driven": [km_driven],
        "fuel_" + fuel: [1],
        "seller_type_" + seller_type: [1],
        "transmission_" + transmission: [1],
        "owner_" + owner: [1],
        "name": [selected_car_name]
    })
    
    if st.sidebar.button("Predict"):
        # Predict car price
        prediction = predict_price(car_data)
        st.sidebar.success(f"The estimated selling price of the car is ₹{prediction[0]}")

if __name__ == "__main__":
    main()
