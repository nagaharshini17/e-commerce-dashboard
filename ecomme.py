import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import os

# Secure Configuration
st.set_page_config(page_title="Dynamic Data Visualization Dashboard", layout="wide")

# Title and Description
st.title("Dynamic Data Visualization Dashboard")
st.markdown("This dashboard allows users to upload their dataset, choose a visualization method, and explore the data dynamically.")

# Upload Dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load Dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")
    st.dataframe(data)

    # Let user select column for analysis
    st.sidebar.header("Select Columns for Analysis")
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) > 0:
        x_axis = st.sidebar.selectbox("Select X-axis Column", options=numeric_columns, index=0)
        y_axis = st.sidebar.selectbox("Select Y-axis Column", options=numeric_columns, index=1)

        # Visualization Options
        st.sidebar.header("Choose Visualization Type")
        visualization_type = st.sidebar.radio(
            "Select the type of visualization:",
            options=["Line Chart", "Bar Chart", "Scatter Plot"],
            index=0
        )

        # Generate Visualization
        st.write("### Visualization")
        if visualization_type == "Line Chart":
            fig = px.line(data, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis} (Line Chart)")
        elif visualization_type == "Bar Chart":
            fig = px.bar(data, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis} (Bar Chart)")
        elif visualization_type == "Scatter Plot":
            fig = px.scatter(data, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis} (Scatter Plot)")

        st.plotly_chart(fig, use_container_width=True)

        # Optional Forecasting Section
        st.sidebar.header("Forecasting (Optional)")
        enable_forecasting = st.sidebar.checkbox("Enable ARIMA Forecasting")

        if enable_forecasting:
            st.write("### Forecasting")

            # Choose column for forecasting
            target_column = st.sidebar.selectbox("Select Column for Forecasting", options=numeric_columns)

            # Forecasting Period
            forecast_period = st.sidebar.slider("Forecast Period (Days)", min_value=7, max_value=30, value=14)

            # Train/Test Split
            forecast_data = data[target_column].dropna()
            train = forecast_data.iloc[:-forecast_period]
            test = forecast_data.iloc[-forecast_period:]

            # Train ARIMA Model
            model = ARIMA(train, order=(5, 1, 0))  # Adjust ARIMA parameters as needed
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_period)

            # Display Forecast Results
            forecast_df = pd.DataFrame({"Actual": test.values, "Forecast": forecast}, index=test.index)
            st.write("#### Forecast vs Actual")
            st.dataframe(forecast_df)

            # Plot Forecast
            fig_forecast = px.line(
                forecast_df,
                title=f"Forecast vs Actual for {target_column}",
                labels={"value": "Values", "index": "Date"}
            )
            fig_forecast.add_scatter(x=test.index, y=test.values, mode="lines", name="Actual")
            fig_forecast.add_scatter(x=test.index, y=forecast, mode="lines", name="Forecast")

            st.plotly_chart(fig_forecast, use_container_width=True)

            # Calculate and Display RMSE
            rmse = np.sqrt(mean_squared_error(test, forecast))
            st.write(f"#### Root Mean Squared Error (RMSE): {rmse:.2f}")

    else:
        st.write("The uploaded dataset does not contain numeric columns for visualization.")
else:
    st.write("Upload a CSV file to start using the dashboard.")