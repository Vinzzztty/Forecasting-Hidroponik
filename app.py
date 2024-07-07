import streamlit as st
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import numpy as np

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
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])  # Ensure datetime format
    return df_prophet


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


def make_predictions(model, future):
    forecast = model.predict(future)
    # Ensure non-negative predictions
    forecast[["yhat", "yhat_lower", "yhat_upper"]] = forecast[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0)
    return forecast


def plot_forecast(model, forecast, periods):
    fig, ax = plt.subplots()
    model.plot(forecast, ax=ax)
    ax.set_title(f"Perkiraan Jumlah Daun untuk {periods} hari ke depan")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Jumlah Daun")
    return fig


def display_hydroponics_info(df, periods):
    st.write("## Informasi Hidroponik")

    if "Label" in df.columns:
        nama_sayuran = df["Label"].unique()
        st.write(f"**Nama Sayuran:** {nama_sayuran}")

    st.write("### Hasil Ramalan")
    st.write(f"- Jumlah hari yang diprediksi: {periods} hari")
    st.write("- Grafik dan tabel informasi hidroponik")


def display_variable_info(df, forecast):
    st.write("## Informasi Variabel")

    # Check if 'ds' column exists in the dataframe
    if "ds" not in df.columns:
        st.write(
            "Dataframe tidak memiliki kolom 'ds'. Pastikan data sudah dimuat dengan benar."
        )
        return

    # Iterate over each variable
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
        st.write(f"### {col}")

        # Plot data
        fig, ax = plt.subplots()
        ax.plot(df["ds"], df[col], label=col)
        ax.set_xlabel("Tanggal")
        ax.set_ylabel(col)
        ax.legend(loc="upper right")
        st.pyplot(fig)

        # Statistics
        st.write("#### Statistik")
        st.write(f"- Nilai Tertinggi: {np.max(df[col]):.2f}")
        st.write(f"- Nilai Terendah: {np.min(df[col]):.2f}")
        st.write(f"- Rata-rata: {np.mean(df[col]):.2f}")

        # Plot forecasted values if available
        if "ds" in forecast.columns:
            st.write("#### Pola Pergerakan selama Ramalan")
            fig, ax = plt.subplots()
            ax.plot(forecast["ds"], forecast[col], label="Ramalan " + col)
            ax.set_xlabel("Tanggal")
            ax.set_ylabel(col)
            ax.legend(loc="upper right")
            st.pyplot(fig)


def main():
    st.markdown("<h1 style='text-align: center;'>Hydrosim</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center;'>Dashboard Ramalan Pertumbuhan Tanaman Selada dan Kualitas Tanaman Selada</h3>",
        unsafe_allow_html=True,
    )

    st.write("**Petunjuk Sebelum Memasukkan File CSV:**")
    st.write(
        "- Pastikan file CSV memiliki kolom yang sesuai (ds, y, hole, temperature, humidity, light, pH, EC, TDS, WaterTemp)."
    )
    st.write(
        "- Format kolom 'ds' harus dalam format datetime (contoh: 'YYYY-MM-DD HH:MM:SS')."
    )
    st.write(
        "- Dataset harus memiliki data yang lengkap untuk akurasi yang lebih baik."
    )

    # Upload CSV file
    st.write("### Unggah File CSV")
    uploaded_file = st.file_uploader(
        "Unggah file CSV untuk dilakukan prediksi", type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Prepare data for Prophet
        df_prophet = prepare_data(df)

        # Initialize and fit the model
        model = initialize_model()
        model = fit_model(model, df_prophet)

        st.write("Hari yang dimasukkan minimal 1 Hari, dan maksimal 365 Hari")

        # User input for number of prediction days
        periods = st.number_input(
            "Masukkan jumlah hari yang ingin diprediksi:",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
        )

        # Create future dataframe
        future = create_future_dataframe(df_prophet, periods=periods)

        # Make predictions
        forecast = make_predictions(model, future)

        # Display forecast table
        st.write(f"## Tabel Prediksi untuk {periods} hari ke depan")
        st.dataframe(
            forecast[
                [
                    "ds",
                    "yhat",
                    "yhat_lower",
                    "yhat_upper",
                    "hole",
                    "temperature",
                    "humidity",
                    "light",
                    "pH",
                    "EC",
                    "TDS",
                    "WaterTemp",
                ]
            ]
        )

        # Plot forecast
        fig = plot_forecast(model, forecast, periods)
        st.pyplot(fig)

        # Display Hydroponics Information
        display_hydroponics_info(df, periods=periods)

        # Display Variable Information
        display_variable_info(df_prophet, forecast)


if __name__ == "__main__":
    main()
