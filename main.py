import streamlit as st
import pandas as pd
from utils import prepare_data, create_future_dataframe
from model import load_model, make_predictions
from visualization import plot_forecast, display_variable_info
import matplotlib.pyplot as plt


def plot_average_leafcount_per_day(df):
    # Ensure 'datetime' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Group by date and calculate the average leaf count
    df["date"] = df["datetime"].dt.date
    avg_leafcount_per_day = df.groupby("date")["LeafCount"].mean().reset_index()

    # Plot the average leaf count per day
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size for better readability
    ax.plot(
        avg_leafcount_per_day["date"],
        avg_leafcount_per_day["LeafCount"],
        marker="o",
        color="green",
    )
    ax.set_title("Rata-Rata Jumlah Daun per Hari")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Rata-Rata Jumlah Daun")
    plt.xticks(rotation=45)

    # Set x-axis major locator to avoid clutter
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    # Improve layout to avoid overlap
    plt.tight_layout()

    st.pyplot(fig)


def main():
    st.markdown(
        """
        <style>
        .header-image {
            width: 100%;
            height: auto;
        }
        </style>
        <h1 style='text-align: center;'>Hydrosim</h1>
        <img src='https://img.idxchannel.com/media/700/images/idx/2022/03/03/Estimasi_biaya_instalasi_hidroponik.jpg' class='header-image'/>
        <h3 style='text-align: center;'>Dashboard Ramalan Pertumbuhan Tanaman Dengan Algoritma Prophet</h3>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Proyek ini bertujuan untuk melakukan **peramalan pertumbuhan tanaman hidroponik** menggunakan data historis mengenai jumlah daun dan berbagai variabel lingkungan seperti suhu, kelembapan, cahaya, pH, dan lainnya. Data yang diupload ini merupakan data minimal 5 hari setelah Tanam (HST)"
    )

    st.markdown("#### **Metodologi dan Kinerja Model:**")

    # Membuat dua kolom
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("- **RMSE (Root Mean Square Error)**: 1.43")
        st.markdown("- **MAE (Mean Absolute Error)**: 1.16")

    with col2:
        st.markdown("- **Jumlah data latih (df_train)**: 8000")
        st.markdown("- **Jumlah data uji (df_test)**: 6400")

    st.markdown("#### **Fitur utama dari aplikasi ini meliputi:**")

    # Membuat dua kolom
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            "- **Unggah file CSV**: Mengunggah file yang berisi data tanaman hidroponik."
        )
        st.markdown(
            "- **Pengolahan data**: Memeriksa dan memproses data untuk memastikan format dan kolom yang diperlukan sudah benar."
        )
        st.markdown(
            "- **Visualisasi data**: Menampilkan grafik rata-rata jumlah daun per hari dan hasil ramalan untuk periode yang akan datang."
        )

    with col4:
        st.markdown(
            "- **Informasi variabel**: Menampilkan informasi dan statistik tentang variabel lingkungan yang mempengaruhi pertumbuhan tanaman."
        )
        st.markdown(
            "- **Peramalan menggunakan model Prophet**: Menggunakan algoritma Prophet untuk meramalkan pertumbuhan daun berdasarkan data historis. Prophet adalah model peramalan yang dikembangkan oleh Facebook, dirancang untuk menangani data yang memiliki pola musiman dan tren yang kuat."
        )

    st.markdown(
        "Harap pastikan bahwa data yang diunggah memenuhi syarat yang disebutkan untuk hasil yang akurat dan bermanfaat."
    )

    st.markdown("### **Petunjuk Sebelum Memasukkan File CSV:**")
    st.markdown(
        "- **Pastikan file CSV Anda memiliki kolom yang diperlukan**, yaitu: `datetime`, `LeafCount`, `hole`, `temperature`, `humidity`, `light`, `pH`, `EC`, `TDS`, dan `WaterTemp`."
    )
    st.markdown(
        "- **Format kolom 'datetime' harus sesuai dengan format datetime standar**, yaitu `YYYY-MM-DD HH:MM:SS`. Contoh: `2024-07-22 14:30:00`."
    )
    st.markdown(
        "- **Pastikan dataset Anda mencakup data yang cukup untuk akurasi ramalan yang optimal**. Data yang dimasukkan harus mencakup **minimal `5` hari dan maksimal `40` hari**."
    )
    st.markdown(
        "- Jika kolom 'datetime' tidak ada, sistem akan mencoba membuatnya dari kolom `day` dan `time`."
    )

    st.write("### Unggah File CSV")
    uploaded_file = st.file_uploader(
        "Unggah file CSV untuk dilakukan prediksi", type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Check if 'datetime' column is present
        if "datetime" not in df.columns:
            st.write(
                "Kolom 'datetime' tidak ditemukan, akan membuat kolom 'datetime' dari kolom 'day' dan 'time'."
            )

            # Tentukan tanggal awal
            start_date = pd.to_datetime("2024-07-01")

            # Ensure 'day' column exists and is of integer type
            if "day" in df.columns:
                df["day"] = df["day"].astype(int)
            else:
                st.error("Kolom 'day' tidak ditemukan pada file CSV.")
                return

            # Ensure 'time' column exists and format it properly
            if "time" in df.columns:
                # Replace dots with colons for time formatting
                df["time"] = df["time"].astype(str).str.replace(".", ":", regex=False)
                # Convert 'time' to datetime.time
                try:
                    df["time"] = pd.to_datetime(df["time"], format="%H:%M").dt.time
                except ValueError as e:
                    st.error(f"Error converting 'time' column: {e}")
                    return
            else:
                st.error("Kolom 'time' tidak ditemukan pada file CSV.")
                return

            # Create 'datetime' column
            df["datetime"] = df.apply(
                lambda row: start_date
                + pd.Timedelta(days=row["day"] - 1)
                + pd.to_timedelta(row["time"].strftime("%H:%M:%S")),
                axis=1,
            )

        # Convert 'datetime' column to datetime format if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            if df["datetime"].isnull().any():
                st.error("Ada nilai yang tidak bisa dikonversi ke format datetime.")
                return

        # Keep only the relevant columns
        important_columns = [
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
        df = df[important_columns]

        plot_average_leafcount_per_day(df)

        df_prophet = prepare_data(df)

        # Load model
        model = load_model("./model/prophet_model.pkl")

        unique_days = df["datetime"].dt.date.nunique()

        st.write(f"Hari yang ada pada dataset: {unique_days} hari")

        future = model.make_future_dataframe(periods=unique_days, freq="D")

        # Assuming the cap value used in training was 18
        future["cap"] = 18

        # Fill in the future DataFrame with the last known values of the regressors from df_test_prophet
        future["hole"] = df_prophet["hole"].iloc[-1]
        future["temperature"] = df_prophet["temperature"].iloc[-1]
        future["humidity"] = df_prophet["humidity"].iloc[-1]
        future["light"] = df_prophet["light"].iloc[-1]
        future["pH"] = df_prophet["pH"].iloc[-1]
        future["EC"] = df_prophet["EC"].iloc[-1]
        future["TDS"] = df_prophet["TDS"].iloc[-1]
        future["WaterTemp"] = df_prophet["WaterTemp"].iloc[-1]

        forecast = model.predict(future)

        # Merge the forecast with the actual test data
        forecast_test = forecast[forecast["ds"].isin(df_prophet["ds"])]
        merged = pd.merge(
            df_prophet,
            forecast_test[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            on="ds",
        )

        st.write("Hari yang dimasukkan minimal 5 Hari, dan maksimal 40 Hari")

        periods = st.number_input(
            "Masukkan jumlah hari yang ingin diprediksi:",
            min_value=5,
            max_value=40,
            value=unique_days,
            step=1,
        )

        future = create_future_dataframe(df_prophet, periods=periods)

        # Assuming the cap value used in training was 18
        future["cap"] = 18

        forecast = make_predictions(model, future)

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

        fig = plot_forecast(model, forecast, periods)
        st.pyplot(fig)

        display_variable_info(df_prophet, forecast)


if __name__ == "__main__":
    main()
