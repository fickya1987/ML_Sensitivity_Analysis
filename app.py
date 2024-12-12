import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Sensitivity Analysis using Machine Learning and GPT-4")

# File uploader
st.subheader("Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display the data
    st.subheader("Dataset")
    st.write(data.head())

    # Select target and features
    st.subheader("Select Target and Features")
    target = st.selectbox("Select the target column", data.columns)
    features = st.multiselect("Select feature columns", [col for col in data.columns if col != target])

    # Select model type
    st.subheader("Select Model Type")
    model_type = st.selectbox("Choose a model", ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest"])

    if target and features:
        X = data[features]
        y = data[target]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the selected model
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Ridge Regression":
            model = Ridge()
        elif model_type == "Lasso Regression":
            model = Lasso()
        elif model_type == "Random Forest":
            model = RandomForestRegressor()

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.subheader("Model Performance")
        st.write(f"Mean Squared Error: {mse}")

        # Sensitivity Analysis
        st.subheader("Sensitivity Analysis")
        if model_type in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
            sensitivities = pd.Series(model.coef_, index=features)
        elif model_type == "Random Forest":
            sensitivities = pd.Series(model.feature_importances_, index=features)

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
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a data analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300
                )
                st.write(response.choices[0].message["content"].strip())
            except Exception as e:
                st.error(f"Error communicating with GPT-4: {e}")
    else:
        st.warning("Please select a target and features for analysis.")
else:
    st.info("Please upload a CSV file to begin.")

