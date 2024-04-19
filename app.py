import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle

def main():
    st.header("Car Price Prediction")
    
   # Load the dataset containing car details
    df = pd.read_csv("CAR DETAILS.csv")
    
    # Function to load the trained model
    @st.cache_data
    def load_model():
        with open("rfmodel.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    
    # Function to preprocess input data
    def preprocess_input(input_data):
        # Your preprocessing steps here, including encoding categorical variables
        split = name.split(" ")
        car_maker = split[0]
        car_model = split[1]

        current_year = 2023
        car_age = current_year-years

        encoder = LabelEncoder()
        # Encode categorical columns in input_df
        categorical_cols = input_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            input_df[col] = label_encoder.fit_transform(input_df[col])
       
    # One-hot encode categorical variables
        pd.get_dummies(input_data, drop_first=True, columns=input_data.columns.difference(['selling_price', 'km_driven', 'year','car_age']))
        return processed_data
    
    # Function to make predictions
    def predict_price(model, input_data):
        # Preprocess input data
        processed_data = preprocess_input(input_data)
        # Make predictions
        prediction = model.predict(processed_data)
        return prediction
    
    # Load the model
    model = load_model()
    
    # User input for car details
    name = st.selectbox("Select Car Name", options=df["name"].unique())
    years = st.selectbox("Select year of model", options=range(1980, 2024))
    km_driven = st.slider('Select km driven', 0.0, 300000.0, step=1000.0)
    fuel = st.selectbox("Select fuel type", options=["Diesel", "Petrol", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("Select seller type", options=["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Select transmission type", options=["Manual", "Automatic"])
    owner = st.selectbox("Select owner type", options=["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
    
    input_data = {
        "name": name,
        "year": years,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner
    }

    # Convert the dictionary into a DataFrame
    input_df = pd.DataFrame(input_data)

    # Predict car price when "Predict" button is clicked
    if st.button('Predict'):
        # Call the predict_price function
        predicted_price = predict_price(model, input_data)
        # Display the predicted price
        st.success(f"Predicted Car Price: {predicted_price}")

if __name__ == "__main__":
    main()
