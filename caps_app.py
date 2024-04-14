import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataframe
df = pd.read_csv("CAR DETAILS.csv")

# Function to preprocess features
def preprocess_features(features):
    df_features = pd.DataFrame(features, index=[0])
    df_features["car_age"] = 2023 - df_features["Year"]
    name = df_features["name"].str.split(" ", expand=True)
    df_features["car_maker"] = name[0]
    df_features["car_model"] = name[1]
    df_features.drop(["name"], axis=1, inplace=True)
    df_features = pd.get_dummies(df_features, drop_first=True)
    return df_features

# Function to predict car price
def predict_price(features):
    df_features = preprocess_features(features)
    prediction = model.predict(df_features)
    return prediction

# Streamlit UI
st.title("Car Price Prediction")

# Sidebar inputs
st.sidebar.header("Enter Car Details")
car_name = st.sidebar.selectbox("Car Name", df["name"].unique())
year = st.sidebar.number_input("Year", 1900, 2022, step=1)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0)
fuel_type = st.sidebar.selectbox("Fuel Type", df["fuel"].unique())
seller_type = st.sidebar.selectbox("Seller Type", df["seller_type"].unique())
transmission = st.sidebar.selectbox("Transmission", df["transmission"].unique())
owner = st.sidebar.selectbox("Owner", df["owner"].unique())

# Get selected car details
car_details = df[(df["name"] == car_name) & 
                 (df["year"] == year) & 
                 (df["fuel"] == fuel_type) & 
                 (df["seller_type"] == seller_type) & 
                 (df["transmission"] == transmission) & 
                 (df["owner"] == owner)]

# Display selected car details
st.sidebar.subheader("Selected Car Details")
if not car_details.empty:
    st.sidebar.write(car_details.iloc[0].to_dict())
else:
    st.sidebar.warning("Car details not found for the selected parameters.")

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
