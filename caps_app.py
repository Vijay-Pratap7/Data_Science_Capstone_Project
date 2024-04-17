import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder

def main():
    st.header("Used Car Price Prediction")

    # File upload
    data = st.file_uploader("Upload a Dataset", type=["csv"])

    if data is not None:
        df = pd.read_csv(data)
        st.dataframe(df.head())

        # Encoding target variable
        encoder = LabelEncoder()
        df["selling_price_encoded"] = encoder.fit_transform(df["selling_price"])

        # UI for user input
        car_name_options = df["name"].unique()
        car_name = st.selectbox("Select Car Name", options=car_name_options)
        split = car_name.split(" ")
        car_maker = split[0]
        car_model = split[1]
        years = st.selectbox("Select year of model", options=range(1980, 2024))
        current_year = 2023
        car_age = current_year - years
        km_driven = st.slider('Select km driven Length', 0.0, 300000.0, step=1000.0)
        fuel_options = ["Diesel", "Petrol", "CNG", "LPG", "Electric"]
        fuel = st.selectbox("Select fuel Width", options=fuel_options)
        seller_type_options = ["Individual", "Dealer", "Trustmark Dealer"]
        seller_type = st.selectbox('Select seller_type Length', options=seller_type_options)
        transmission_options = ["Manual", "Automatic"]
        transmission = st.selectbox('Select transmission Width', options=transmission_options)
        owner_options = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
        owner = st.selectbox('Select owner Width', options=owner_options)

        test_data = [[car_maker, car_model, car_age, km_driven, fuel, seller_type, transmission, owner]]
        st.write('Test_Data:', test_data)

        if st.button('Predict'):
            input_data = {
                "car_maker": [car_maker],
                "car_model": [car_model],
                "car_age": [car_age],
                'km_driven': [km_driven],
                'fuel': [fuel],
                'seller_type': [seller_type],
                'transmission': [transmission],
                'owner': [owner]
            }
            input_df = pd.DataFrame(input_data)

            # Load the best model
            with open('rfmodel.pkl', 'rb') as file:
                best_model = pickle.load(file)

            # Predict on input data
            prediction = best_model.predict(input_df)

            if prediction[0] < 0:
                st.error("There were inaccuracies in the details entered by you.")
            else:
                st.success(f"Predicted Selling Price: {round(prediction[0], 2)}")

if __name__ == "__main__":
    main()
