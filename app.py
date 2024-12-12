import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Add company logo
st.image("pelindo_logo.jfif", use_column_width=True)

st.title("Pelindo-TKMP AI Sensitivity Analysis")

# File uploader
st.subheader("Unggah Dataset")
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

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

    st.write("Kolom Kategorikal:", categorical_cols)
    st.write("Kolom Numerik:", numerical_cols)

    # Select target and features
    st.subheader("Pilih Target dan Fitur")
    if numerical_cols:
        target = st.selectbox("Pilih kolom target", numerical_cols)
        features = st.multiselect("Pilih kolom fitur", [col for col in numerical_cols if col != target])
    else:
        st.error("Tidak ada kolom numerik yang tersedia untuk dijadikan target. Mohon periksa dataset Anda.")
        target, features = None, None

    # Select model type
    st.subheader("Pilih Jenis Model")
    model_type = st.selectbox("Pilih model", ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest"])

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

            # Performance metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            st.subheader("Kinerja Model")
            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            st.write(f"R-squared (R²): {r2:.4f}")

            st.write("Penjelasan:")
            st.write("- **MSE** mengukur rata-rata kesalahan kuadrat antara prediksi dan nilai aktual. Semakin kecil nilainya, semakin baik.")
            st.write("- **MAE** menunjukkan rata-rata kesalahan absolut, memberikan gambaran tentang rata-rata penyimpangan prediksi.")
            st.write("- **RMSE** memberikan kesalahan dalam skala yang sama dengan target, sangat berguna untuk interpretasi langsung.")
            st.write("- **R²** menunjukkan seberapa baik model menjelaskan variasi dalam data. Nilai mendekati 1 menunjukkan model yang baik.")

            # Sensitivity Analysis
            st.subheader("Analisis Sensitivitas")
            if model_type in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
                regressor = model.named_steps["regressor"]
                sensitivities = pd.Series(regressor.coef_, index=features)
            elif model_type == "Random Forest":
                regressor = model.named_steps["regressor"]
                sensitivities = pd.Series(regressor.feature_importances_, index=features)

            sensitivities = sensitivities.abs().sort_values(ascending=False)

            st.write("Sensitivitas Fitur:")
            st.bar_chart(sensitivities)

            # Example insights
            st.write("Contoh Hasil Analisis:")
            for feature, sensitivity in sensitivities.items():
                st.write(f"- **{feature}:** Sensitivitas {sensitivity:.4f}. Interpretasi: Hubungan fitur ini dengan target dapat menunjukkan dampak langsung atau tidak langsung pada variabel target.")

            # GPT-4 Analysis
            st.subheader("Analisis GPT-4")
            if st.button("Analisis dengan GPT-4"):
                prompt = (
                    f"Dataset ini memiliki kolom: {list(data.columns)}. Kolom target adalah '{target}', dan kolom fitur adalah {features}. "
                    f"Model {model_type} memiliki MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, dan R²: {r2:.4f}. "
                    f"Sensitivitas fitur adalah sebagai berikut: {sensitivities.to_dict()}. "
                    "Berikan wawasan tentang hubungan fitur dengan target, sertakan juga bagaimana perubahan pada setiap fitur (misalnya, kenaikan 10%) akan memengaruhi target dalam konteks sensitivitasnya. "
                    "Berikan rekomendasi bisnis berdasarkan analisis ini."
                )

                # GPT-4 API call
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "Anda adalah ahli analisis data."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2048
                    )
                    st.write(response.choices[0].message["content"].strip())
                except Exception as e:
                    st.error(f"Error berkomunikasi dengan GPT-4: {e}")
        except Exception as e:
            st.error(f"Error selama pelatihan atau analisis model: {e}")
    else:
        st.warning("Mohon pilih target dan fitur untuk analisis.")
else:
    st.info("Unggah file CSV untuk memulai.")



