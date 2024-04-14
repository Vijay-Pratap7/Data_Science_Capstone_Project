import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import streamlit as st

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv("CAR DETAILS.csv")
    return df

# Preprocess the data
def preprocess_data(df):
    df.drop_duplicates(inplace=True)
    df["car_age"] = 2023 - df["year"]
    name = df["name"].str.split(" ", expand=True)
    df["car_maker"] = name[0]
    df["car_model"] = name[1]
    df.drop(["name"], axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True, columns=df.columns.difference(['selling_price', 'km_driven', 'year', 'car_age']))
    return df

# Split data into features and target
def split_data(df):
    X = df.drop("selling_price", axis=1)
    y = df["selling_price"]
    return X, y

# Train and evaluate models
def train_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest Regressor": RandomForestRegressor()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MAE": mae, "MSE": mse, "R2 Score": r2}
    return results

# Save the best model
def save_model(model):
    with open('rfmodel.pkl', 'wb') as file:
        pickle.dump(model, file)

# Main function
def main():
    st.title("Used Car Price Prediction")
    st.sidebar.title("Model Evaluation")

    # Load data
    df = load_data()
    df = preprocess_data(df)

    # Split data
    X, y = split_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Train and evaluate models
    results = train_evaluate_models(X_train, X_test, y_train, y_test)

    # Display results
    st.write("### Model Evaluation Results")
    for model, metrics in results.items():
        st.write(f"**{model}**")
        st.write("MAE:", metrics["MAE"])
        st.write("MSE:", metrics["MSE"])
        st.write("R2 Score:", metrics["R2 Score"])
        st.write("---")

    # Save the best model
    best_model = RandomForestRegressor()
    best_model.fit(X_train, y_train)
    save_model(best_model)

if __name__ == "__main__":
    main()
