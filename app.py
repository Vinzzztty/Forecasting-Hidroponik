import streamlit as st
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
register_matplotlib_converters()

df2 = pd.read_csv("https://raw.githubusercontent.com/Vinzzztty/Forecasting-Hidroponik/main/dataset/Dataset_v2.csv")

df2 = df2.drop(columns=['Jam', 'Label'])

# Define features and target
features = ['temperature', 'humidity', 'light', 'pH', 'EC', 'TDS', 'WaterTemp']
X = df2[features]
y = df2['Pattern']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Gradient Boosting Classifier with specified parameters
model = GradientBoostingClassifier(learning_rate=0.1, max_depth=10)

# Train the model
model.fit(X_train, y_train)

# Predict function
def predict_pattern(model, features):
    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(f"Enter {feature}", min_value=float(X[feature].min()), max_value=float(X[feature].max()), value=float(X[feature].mean()))
    input_data = pd.DataFrame([user_input])
    prediction = model.predict(input_data)
    if prediction == 1:
        return "Pola Biasa"
    elif prediction == 2:
        return "Pola Ideal"
    elif prediction == 3:
        return "Pola Berlebihan"

# Function to prepare data for Prophet
def prepare_data(df):
    df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure datetime column is datetime type
    return df[["datetime", "LeafCount"]].rename(columns={"datetime": "ds", "LeafCount": "y"})

# Function to forecast using Prophet
def forecast_prophet(df, num_days):
    prophet_model = Prophet()
    prophet_model.fit(prepare_data(df))
    future_dates = prophet_model.make_future_dataframe(periods=num_days, freq="D")
    forecast = prophet_model.predict(future_dates)
    forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(num_days)
    forecast_table = forecast_table.rename(
        columns={
            "ds": "Tanggal", 
            "yhat": "Jumlah Daun", 
            "yhat_lower": "Jumlah Daun Terendah", 
            "yhat_upper": "Jumlah Daun Tertinggi"
        }
    )
    return forecast_table, forecast

# Function to generate the Streamlit app
def main():
    st.markdown("<h1 style='text-align: center;'>Hydrosim</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Dashboard Ramalan Pertumbuhan Tanaman Selada dan Kualitas Tanaman Selada</h3>", unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        # Example: Load your dataframe here
        df = pd.read_csv(
            "https://raw.githubusercontent.com/Vinzzztty/Forecasting-Hidroponik/main/dataset/dataset_final.csv",
            parse_dates=['datetime']  # Parse datetime column
        )  # Replace with your data loading logic
        return df

    df = load_data()

    # User input for forecasting in main body
    num_days_to_forecast = st.number_input("Number of days to forecast", min_value=1, max_value=30, value=5)

    # Forecasting with Prophet
    st.markdown(f"<h3 style='text-align: center;'>Forecasting for the next {num_days_to_forecast} days</h3>", unsafe_allow_html=True)

    forecast_df, full_forecast = forecast_prophet(df, num_days_to_forecast)

    # Plot the forecast
    fig = plt.figure(figsize=(12, 6))
    
    # Plot original data (blue)
    plt.plot(df['datetime'], df['LeafCount'], label='Original Data', color='blue')

    # Plot forecasted values (green)
    plt.plot(full_forecast['ds'], full_forecast['yhat'], label='Forecast', color='green')

    # Fill uncertainty area
    plt.fill_between(full_forecast['ds'], full_forecast['yhat_lower'], full_forecast['yhat_upper'], color='lightgreen', alpha=0.4)
    
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah Daun')
    plt.title('Ramalan dengan Prophet')
    plt.legend()
    st.pyplot(fig)

    st.markdown("<p style='text-align: center;'>Gambar Visualisasi Peralaman Pertumbuhan Tanaman Selada</p>", unsafe_allow_html=True)

    # Show the forecasted results in a table
    st.write(forecast_df.set_index("Tanggal"))


    st.markdown("<h3 style='text-align: center;'>Pola Pertumbuhan Tanaman Selada</h3>", unsafe_allow_html=True)
    st.markdown("Pola 1: Biasa\nPola 2: Ideal\nPola 3: Berlebihan")
    st.markdown("### Masukkan Nilai Parameter Untuk Prediksi Pola Pertumbuhan Tanaman Selada")

    predicted_pattern = predict_pattern(model, features)
    st.markdown(f"### Predicted Pattern: {predicted_pattern}")

if __name__ == "__main__":
    main()
