import streamlit as st
import pandas as pd
import pickle

# Load the saved trained ML model
with open('rfmodel.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Load the dataset
df = pd.read_csv("CAR DETAILS.csv")

# Function to predict car price
def predict_price(model, input_data):
    return model.predict(input_data)

# Streamlit app
def main():
    st.title("Car Price Prediction App")

    # Dropdown to select car name
    selected_car = st.selectbox("Select Car Name", df["name"])

    # Display selected car details
    selected_car_details = df[df["name"] == selected_car].iloc[0]
    st.write("Selected Car Details:")
    st.write(selected_car_details)

    # Prepare input data for prediction
    input_data = selected_car_details.drop(["name", "selling_price"]).to_numpy().reshape(1, -1)

    # Predict car price
    if st.button("Predict Price"):
        predicted_price = predict_price(best_model, input_data)
        st.success(f"Predicted Selling Price: {predicted_price[0]}")

if __name__ == "__main__":
    main()
