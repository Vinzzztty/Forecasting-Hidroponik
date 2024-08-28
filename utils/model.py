from prophet import Prophet
import pandas as pd
import joblib


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


def load_model(model_path):
    model_loaded = joblib.load(model_path)

    return model_loaded


def create_future_dataframe(df_test, periods):
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


def make_predictions(model, future):
    forecast = model.predict(future)
    forecast[["yhat", "yhat_lower", "yhat_upper"]] = forecast[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0)
    return forecast

    return forecast
