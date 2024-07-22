from prophet import Prophet


def initialize_model():
    model = Prophet()
    model.add_regressor("hole")
    model.add_regressor("temperature")
    model.add_regressor("humidity")
    model.add_regressor("light")
    model.add_regressor("pH")
    model.add_regressor("EC")
    model.add_regressor("TDS")
    model.add_regressor("WaterTemp")
    return model


def fit_model(model, df_train):
    model.fit(df_train)
    return model


def make_predictions(model, future):
    forecast = model.predict(future)
    forecast[["yhat", "yhat_lower", "yhat_upper"]] = forecast[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0)
    return forecast
