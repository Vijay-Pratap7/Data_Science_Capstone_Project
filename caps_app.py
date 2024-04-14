import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Function to load data
@st.cache
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to preprocess data
def preprocess_data(df):
    df.drop_duplicates(inplace=True)
    if 'name' in df.columns:
        df["car_age"] = 2023 - df["year"]
        name = df["name"].str.split(" ", expand=True)
        df["car_maker"] = name[0]
        df["car_model"] = name[1]
        df.drop(["name"], axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    return df

# Function to train model
def train_model(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

# Function to predict price
def predict_price(model, features):
    prediction = model.predict(features)
    return prediction

# Main function
def main():
    st.title("Car Price Prediction")

    # Upload file
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is not None:
        df = load_data(file)
        df = preprocess_data(df)

        # Select target variable
        st.subheader("Select target variable")
        target_col = st.selectbox("Select target variable", df.columns)

        # Train model
        model = train_model(df, target_col)

        # Input features
        st.subheader("Enter Car Details to Predict Price")
        features = {}
        for col in df.columns:
            if col != target_col:
                features[col] = st.number_input(f"Enter {col}", min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(df[col].mean()))

        if st.button("Predict"):
            input_features = pd.DataFrame(features, index=[0])
            prediction = predict_price(model, input_features)
            st.success(f"Predicted Selling Price: {prediction[0]}")

if __name__ == "__main__":
    main()
