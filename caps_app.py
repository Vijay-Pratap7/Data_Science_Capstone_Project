import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained Random Forest model
with open('rfmodel.pkl', 'rb') as file:
    model = pickle.load(file)
    
data = st.file_uploader("Upload a Dataset", type = ["csv"])

    if data is not None:
      df = pd.read_csv(data)
      st.dataframe(df.head())
# Define a function to preprocess user input
def preprocess_input(year, km_driven, owner, fuel, seller_type, transmission, name):
    name = df["name"].str.split(" ", expand = True)
    df["car_maker"] = name[0]
    df["car_model"] = name[1]
    
    df["car_age"] = 2023 - df["year"]
    df.drop(["name"], axis=1, inplace=True)
    
    # One-hot encode categorical variables
    data = pd.get_dummies((df, drop_first=True, columns=df.columns.difference(['selling_price', 'km_driven', 'year','car_age']))
    encoder = LabelEncoder()
    data_encoded = data.apply(encoder.fit_transform)
    # Ensure the input DataFrame has the same columns as the training data
    # Missing columns will be added with values of 0
    X_columns = model.named_steps['preprocessor'].transformers_[0][2] + \
                model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out().tolist()
    data = data.reindex(columns=X_columns, fill_value=0)
    return data

# Define the Streamlit app
def main():
    st.title("Car Price Prediction")

    # User input fields
    name = st.selectbox("Select Car Name", options = df["name"].unique())
      st.write(name)
      split = name.split(" ")
      car_maker = split[0]
      car_model = split[1]
    year = st.number_input("Year of manufacture", min_value=1900, max_value=2023, value=2010)
    km_driven = st.number_input("Kilometers driven", min_value=0, value=50000)
    owner = st.selectbox("Owner type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
    fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("Seller type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission type", ["Manual", "Automatic"])

    # Preprocess user input
    input_data = preprocess_input(year, km_driven, owner, fuel, seller_type, transmission, name)

    # Predict car price
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.success(f"The estimated price of the car is {prediction[0]:,.2f} INR")

if __name__ == "__main__":
    main()
