import streamlit as st
from prophet import Prophet
import pandas as pd
import numpy as np


# Function to prepare data for Prophet
def prepare_data(df):
    return df[["datetime", "LeafCount"]].rename(
        columns={"datetime": "ds", "LeafCount": "y"}
    )


# Function to forecast using Prophet
def forecast_prophet(df, num_days):
    prophet_model = Prophet()
    prophet_model.fit(prepare_data(df))
    future_dates = prophet_model.make_future_dataframe(periods=num_days, freq="D")
    forecast = prophet_model.predict(future_dates)
    forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(num_days)
    return forecast_table


# Function to generate the Streamlit app
def main():
    st.title("Prophet Forecasting Dashboard")

    # Sidebar - User input for forecasting
    st.sidebar.header("Input Parameters")
    num_days_to_forecast = st.sidebar.number_input(
        "Number of days to forecast", min_value=1, max_value=30, value=5
    )

    @st.cache
    def load_data():
        # Example: Load your dataframe here
        df = pd.read_csv(
            "./dataset/DataTrainSIOHIFull.csv"
        )  # Replace with your data loading logic
        return df

    df = load_data()

    # Show a preview of the data
    st.subheader("Data Preview")
    st.write(df.head())

    # Forecasting with Prophet
    st.subheader(f"Forecasting for the next {num_days_to_forecast} days")

    forecast_df = forecast_prophet(df, num_days_to_forecast)

    # Show the forecasted results in a table
    st.write(forecast_df.set_index("ds"))


if __name__ == "__main__":
    main()
