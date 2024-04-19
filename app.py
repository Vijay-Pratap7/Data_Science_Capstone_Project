import sklearn
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import pandas as pd

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
      df.drop(["name"], axis=1, inplace=True)

      years = st.selectbox("Select year of model", options=range(1980, 2024))

      current_year = 2023
      car_age = current_year-years

      km_driven = st.slider('Select km driven Length', 0.0, 300000.0, step = 1000.0)

      fuel = st.selectbox("Select fuel type", options=["Diesel", "Petrol", "CNG", "LPG", "Electric"])

      seller_type = st.selectbox("Select seller type", options=["Individual", "Dealer", "Trustmark Dealer"])

      transmission = st.selectbox("Select transmission type", options=["Manual", "Automatic"])

      owner = st.selectbox("Select owner type", options=["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

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

        input_df = pd.DataFrame(input_data)
        encoder = LabelEncoder()
        encoded_columns = ['fuel', 'seller_type', 'transmission', 'owner']

        for i in encoded_columns:
           input_df[i] = encoder.fit_transform(input_df[i])

    # One-hot encode categorical variables
        input_df = pd.get_dummies(input_df, drop_first=True, columns = ["car_maker", "car_model"])

        # Update the file path to reflect the correct location in the Streamlit cloud
        pkl_file_path = "rfmodel.pkl"

        # Load the pickle file
        with open(pkl_file_path, "rb") as file:
          model = pickle.load(file)


        predictions = model.predict(final_data)
        
        if predictions<0:
            st.success("There were inaccuracies in the details entered by you.")
        else:
            st.success(round(predictions[0]))

        st.success(predictions[0])


if __name__ == "__main__":
    main()
