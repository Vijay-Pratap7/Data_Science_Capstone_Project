import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataframe
df = pd.read_csv("CAR DETAILS.csv")

# Check dataframe columns
st.write("DataFrame Columns:", df.columns)

# Function to preprocess features
def preprocess_features(features):
    df_features = pd.DataFrame(features, index=[0])
    df_features["car_age"] = 2023 - df_features["year"]
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
# Check if car_name exists in dataframe
if car_name in df["name"].unique():
    selected_car = df[df["name"] == car_name]
    if not selected_car.empty:
        selected_car = selected_car.iloc[0]  # Get the first row with the selected car name
        year = selected_car["year"]
        km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0)
        fuel_type = st.sidebar.selectbox("Fuel Type", df["fuel"].unique())
        seller_type = st.sidebar.selectbox("Seller Type", df["seller_type"].unique())
        transmission = st.sidebar.selectbox("Transmission", df["transmission"].unique())
        owner = st.sidebar.selectbox("Owner", df["owner"].unique())

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
    else:
        st.warning("Selected car details not found in the dataframe.")
else:
    st.warning("Selected car name not found in the dataframe.")
