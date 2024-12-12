import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

openai.api_key = openai_api_key

st.title("Sensitivity Analysis using Machine Learning and GPT-4")

# File uploader
st.subheader("Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display the data
    st.subheader("Dataset")
    st.write(data.head())

    # Handle non-numeric data and convert numeric strings to floats
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = data[col].str.replace(',', '').str.replace(' ', '').astype(float)
            except ValueError:
                pass  # Leave non-convertible columns as is

    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

    st.write("Categorical Columns:", categorical_cols)
    st.write("Numerical Columns:", numerical_cols)

    # Select target and features
    st.subheader("Select Target and Features")
    if numerical_cols:
        target = st.selectbox("Select the target column", numerical_cols)
        features = st.multiselect("Select feature columns", [col for col in numerical_cols if col != target])
    else:
        st.error("No numerical columns available for target selection. Please check your dataset.")
        target, features = None, None

    # Select model type
    st.subheader("Select Model Type")
    model_type = st.selectbox("Choose a model", ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest"])

    if target and features:
        try:
            X = data[features]
            y = data[target]

            # Handle missing values
            X = X.fillna(0)  # Replace NaN with 0
            y = y.fillna(0)

            # Preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), features)
                ],
                remainder='passthrough'
            )

            # Model pipeline
            if model_type == "Linear Regression":
                model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])
            elif model_type == "Ridge Regression":
                model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", Ridge())])
            elif model_type == "Lasso Regression":
                model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", Lasso())])
            elif model_type == "Random Forest":
                model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor())])

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            st.subheader("Model Performance")
            st.write(f"Mean Squared Error: {mse}")

            # Sensitivity Analysis
            st.subheader("Sensitivity Analysis")
            if model_type in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
                regressor = model.named_steps["regressor"]
                sensitivities = pd.Series(regressor.coef_, index=features)
            elif model_type == "Random Forest":
                regressor = model.named_steps["regressor"]
                sensitivities = pd.Series(regressor.feature_importances_, index=features)

            sensitivities = sensitivities.abs().sort_values(ascending=False)

            st.write("Feature Sensitivities:")
            st.bar_chart(sensitivities)

            # GPT-4 Analysis
            st.subheader("GPT-4 Analysis")
            if st.button("Analyze with GPT-4"):
                prompt = (
                    f"The dataset contains the following columns: {list(data.columns)}. The target column is '{target}', and the feature columns are {features}. "
                    f"The {model_type} model achieved a Mean Squared Error of {mse}. The sensitivities of the features are as follows: {sensitivities.to_dict()}. "
                    "Provide insights about the relationships between features and the target, and suggest potential improvements or considerations for this analysis."
                )

                # GPT-4 API call
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a data analysis expert."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2048
                    )
                    st.write(response.choices[0].message["content"].strip())
                except Exception as e:
                    st.error(f"Error communicating with GPT-4: {e}")
        except Exception as e:
            st.error(f"Error during model training or analysis: {e}")
    else:
        st.warning("Please select a target and features for analysis.")
else:
    st.info("Please upload a CSV file to begin.")


