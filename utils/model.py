from prophet import Prophet
import joblib


def load_model(model_path):
    model_loaded = joblib.load(model_path)

    return model_loaded
