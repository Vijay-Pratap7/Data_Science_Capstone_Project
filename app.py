import sklearn
import streamlit as st
import scipy
import scipy.stats
import itertools
import pickle

# Importing Libraries for EDA
import pandas as pd
import numpy as np

def data_preprocess():
    encoder = LabelEncoder()
    df1 = input_df.apply(encoder.fit_transform)
    return df1
    # One-hot encode categorical variables
    final_data = pd.get_dummies(df1, drop_first=True, columns=df1.columns.difference(['selling_price', 'km_driven', 'year','car_age']))
def main():
    st.header("Car Price Prediction")
    data = st.file_uploader("Upload a Dataset", type = ["csv"])

    if data is not None:
      df = pd.read_csv(data)
      st.dataframe(df.head())

      name = st.selectbox("Select Car Name", options = df["name"].unique())
      st.write(name)
      split = name.split(" ")
      car_maker = split[0]
      car_model = split[1]


      option = list(itertools.chain(range(1980, 2024, 1)))

      years = st.selectbox("Select year of model", options = option)
      st.write(years)

      current_year = 2023
      car_age = current_year-years

      km_driven = st.slider('Select km driven Length', 0.0, 300000.0, step = 1000.0)

      fuel_options = ["Diesel", "Petrol", "CNG", "LPG", "Electric"]
      fuel = st.selectbox("Select fuel Width", options = fuel_options)

      seller_type_options = ["Individual", "Dealer", "Trustmark Dealer"]
      seller_type = st.selectbox('Select seller_type Length', options = seller_type_options)

      transmission_options = ["Manual", "Automatic"]
      transmission = st.selectbox('Select transmission Width', options = transmission_options)

      owner_options = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
      owner = st.selectbox('Select owner Width', options = owner_options)

      test  = [[ name, years, km_driven, fuel, seller_type, transmission, owner]]
      st.write('Test_Data', test)


      if st.button('Predict', key = "int"):
        input_data = {"car_maker": [car_maker],
                    "car_model": [car_model],
                    "car_age":[car_age],
                    'km_driven': [km_driven],
                    'fuel': [fuel],
                    'seller_type': [seller_type],
                    'transmission': [transmission],
                    'owner': [owner]}

        input_df = pd.DataFrame(final_data)
        data_preprocess()
        # Update the file path to reflect the correct location in the Streamlit cloud
        pkl_file_path = "rfmodel.pkl"

        # Load the pickle file
        with open(pkl_file_path, "rb") as file:
          model = pickle.load(file)


        predictions = model.predict(input_df)
        
        if predictions<0:
            st.success("There were inaccuracies in the details entered by you.")
        else:
            st.success(round(predictions[0]))

      #  st.success(predictions[0])


if __name__ == "__main__":
    main()
