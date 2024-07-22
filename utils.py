import pandas as pd
import streamlit as st
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


@st.cache_data
def load_data():
    df_train = pd.read_csv(
        "https://raw.githubusercontent.com/Vinzzztty/Forecasting-Hidroponik/main/dataset/dataset_train_final.csv"
    )
    df_test = pd.read_csv(
        "https://raw.githubusercontent.com/Vinzzztty/Forecasting-Hidroponik/main/dataset/dataset_test_final.csv"
    )
    return df_train, df_test


def prepare_data(df):
    df_prophet = df[
        [
            "datetime",
            "LeafCount",
            "hole",
            "temperature",
            "humidity",
            "light",
            "pH",
            "EC",
            "TDS",
            "WaterTemp",
        ]
    ].copy()
    df_prophet.rename(columns={"datetime": "ds", "LeafCount": "y"}, inplace=True)
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
    return df_prophet


def create_future_dataframe(df_test, periods=30):
    future_dates = pd.date_range(start=df_test["ds"].max(), periods=periods, freq="D")
    last_row = df_test.iloc[-1]

    future = pd.DataFrame({"ds": future_dates})
    for col in [
        "hole",
        "temperature",
        "humidity",
        "light",
        "pH",
        "EC",
        "TDS",
        "WaterTemp",
    ]:
        future[col] = last_row[col]
    return future
