import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Function to load and preprocess data
def load_data(file):
    df = pd.read_csv(file)
    df.drop_duplicates(inplace=True)
    df["car_age"] = 2023 - df["year"]
    name = df["name"].str.split(" ", expand=True)
    df["car_maker"] = name[0]
    df["car_model"] = name[1]
    df.drop(["name"], axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    return df

# Function to train and evaluate models
def train_and_evaluate(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "KNeighbors Regressor": KNeighborsRegressor(),
        "Random Forest Regressor": RandomForestRegressor()
    }

    best_model = None
    best_score = -float("inf")
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        if test_score > best_score:
            best_model = model
            best_score = test_score

    return best_model, X_train, X_test, y_train, y_test

# Function to save the best model
def save_model(model):
    with open('best_model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Function to load the best model
def load_model():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to predict on sample data
def predict_sample_data(model, df_sample):
    X_sample = df_sample.drop(target_col, axis=1)
    y_sample = df_sample[target_col]
    y_pred_sample = model.predict(X_sample)
    return y_sample, y_pred_sample

# Main function
def main():
    st.title("Car Price Prediction")

    # Upload file
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is not None:
        df = load_data(file)

        # Select target variable
        st.subheader("Select target variable")
        target_col = st.selectbox("Select target variable", df.columns)

        # Train and evaluate models
        st.subheader("Train and Evaluate Models")
        best_model, X_train, X_test, y_train, y_test = train_and_evaluate(df, target_col)

        # Save the best model
        save_model(best_model)

        # Load the best model
        model = load_model()

        # Create a sample random dataframe
        df_sample = df.sample(20, random_state=42)

        # Predict on sample data
        y_sample, y_pred_sample = predict_sample_data(model, df_sample)

        # Display scores
        st.subheader("Scores for main data:")
        st.write("Training Score:", model.score(X_train, y_train))
        st.write("Testing Score:", model.score(X_test, y_test))
        st.write("R2 Score:", r2_score(y_test, model.predict(X_test)))
        st.write("MAE:", mean_absolute_error(y_test, model.predict(X_test)))
        st.write("MSE:", mean_squared_error(y_test, model.predict(X_test)))

        st.subheader("Scores for sample data:")
        st.write("R2 Score:", r2_score(y_sample, y_pred_sample))
        st.write("MAE:", mean_absolute_error(y_sample, y_pred_sample))
        st.write("MSE:", mean_squared_error(y_sample, y_pred_sample))

if __name__ == "__main__":
    main()
