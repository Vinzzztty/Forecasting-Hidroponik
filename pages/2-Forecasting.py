import streamlit as st
import pandas as pd
from utils import model, visualization, cek_optimization
import matplotlib.pyplot as plt
import time
import warnings

# GLOBAL VARIABLE
MAX_DAY = 40


def set_page_config():
    """Set the initial page configuration."""
    st.set_page_config(
        page_icon="https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/logo_hijau.png?raw=true",
        page_title="Hydrosim - Forecasting",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_css():
    """Inject custom CSS for styling."""
    st.markdown(
        """
        <style>
        /* Styling the header image */
        .header-image {
            width: 100%;
            height: auto;
        }
        
        /* Change the background color of the sidebar */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with navigation."""
    with st.sidebar:
        st.markdown(
            "![Logo](https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/new_hijau.png?raw=true)"
        )


def main():
    set_page_config()
    inject_custom_css()

    render_sidebar()

    st.title("Welcome to Forecasting Page")

    # Display the options to the user
    option = st.radio(
        "Pilih metode input data:", ("Unggah file CSV", "Gunakan contoh file CSV")
    )

    if option == "Unggah file CSV":
        # File uploader
        uploaded_file = st.file_uploader(
            "Unggah file CSV untuk dilakukan prediksi", type=["csv"]
        )

        if uploaded_file is not None:

            df = pd.read_csv(uploaded_file)

            # check if 'datetime' column is not present
            if "datetime" not in df.columns:
                st.info(
                    "Kolom 'datetime' tidak ditemukan, akan membuat kolom 'datetime' dari kolom 'day' dan 'time' secara otomatis!."
                )

                # Inisialisasi start_date
                # now = datetime.now().strftime("%Y-%m-%d")
                start_date = pd.to_datetime("2024-07-01")

                # Ensure 'day' column exists and is of integer type
                if "day" in df.columns:
                    df["day"] = df["day"].astype(int)
                else:
                    st.error("Kolom 'day' tidak ditemukan pada file CSV.")

                # Ensure 'time' column exists and format it properly
                if "time" in df.columns:
                    df["time"] = df["time"].apply(lambda x: "{:.2f}".format(x))

                    df["time"] = pd.to_datetime(df["time"], format="%H.%M").dt.time

                    # Replace dots with colons for time formatting
                    df["time"] = df["time"].astype(
                        str
                    )  # .str.replace(".", ":", regex=False)

                    df["datetime"] = df.apply(
                        lambda row: start_date
                        + pd.Timedelta(days=row["day"] - 1)
                        + pd.to_timedelta(row["time"]),
                        axis=1,
                    )

                    df = df.drop_duplicates(subset=["day", "time", "LeafCount"])

                    df.set_index("datetime", inplace=True)
                    df = df.sort_index()

                else:
                    st.error("Kolom 'time' tidak ditemukan pada file CSV.")
                    return

                df["datetime"] = df.apply(
                    lambda row: start_date
                    + pd.Timedelta(days=row["day"] - 1)
                    + pd.to_timedelta(row["time"]),
                    axis=1,
                )

            # Convert 'datetime'column to datetime format if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

                if df["datetime"].isnull().any():
                    st.error(
                        "⚠️ Ada nilai yang tidak bisa dikonversi ke format datetime."
                    )
                    return

            # important column
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

            st.markdown("### 📊 Data tanaman yang di Upload")

            st.dataframe(df)

            df_prophet = model.prepare_data(df)

            # Load Model
            models = model.load_model("./model/prophet_model.pkl")

            unique_days = df["datetime"].dt.date.nunique()

            st.info(f"🗓️ Total hari setelah di Tanam: {unique_days} hari")

            with st.spinner(text="⏳ Sedang menganalisis..."):
                time.sleep(2)
                # st.success("Done")

            future = models.make_future_dataframe(periods=unique_days, freq="D")

            future["cap"] = 18

            future["hole"] = df_prophet["hole"].iloc[-1]
            future["temperature"] = df_prophet["temperature"].iloc[-1]
            future["humidity"] = df_prophet["humidity"].iloc[-1]
            future["light"] = df_prophet["light"].iloc[-1]
            future["pH"] = df_prophet["pH"].iloc[-1]
            future["EC"] = df_prophet["EC"].iloc[-1]
            future["TDS"] = df_prophet["TDS"].iloc[-1]
            future["WaterTemp"] = df_prophet["WaterTemp"].iloc[-1]

            forecast = models.predict(future)

            forecast_test = forecast[forecast["ds"].isin(df_prophet["ds"])]

            merged = pd.merge(
                df_prophet,
                forecast_test[["ds", "yhat", "yhat_lower", "yhat_upper"]],
                on="ds",
            )

            max_periods = MAX_DAY - unique_days

            periods = st.slider(
                "⏳ Pilih hari untuk Forecasting pertumbuhan daun",
                min_value=unique_days,
                max_value=max_periods,
                step=1,
            )

            future = model.create_future_dataframe(df_prophet, periods=periods)

            future["cap"] = 18

            forecast = model.make_predictions(models, future)

            st.markdown(""" --- """)
            st.markdown(f"### 📈 Hasil Forecasting untuk {periods} Hari Ke Depan")

            st.write(f"🔍 Visualisasi Prediksi")
            fig = visualization.plot_forecast(forecast, periods)

            # st.pyplot(fig)
            st.plotly_chart(fig)

            col1, col2 = st.columns([6, 4])

            with col1:
                max_forecasted_leaf_count = forecast["yhat"].max()

                if max_forecasted_leaf_count <= 6:
                    image_path = "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/early_leaf.png?raw=true"
                elif max_forecasted_leaf_count <= 12:
                    image_path = "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/mid_leaf.png?raw=true"
                else:
                    image_path = "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/high_leaf.png?raw=true"

                # Center the image using HTML and display it
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <img src="{image_path}" alt="Prediksi Jumlah Daun Selada" style="width: 50%;">
                        <p>Prediksi Jumlah Daun Selada</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.write(f"📋 Tabel Prediksi")
                st.dataframe(
                    forecast[
                        [
                            "ds",
                            "yhat",
                            "yhat_lower",
                            "yhat_upper",
                        ]
                    ]
                )

            warnings.filterwarnings("ignore")

            st.markdown(f"#### Kualitas Tanaman Selada")

            # Display loading spinner while the model is being loaded
            with st.spinner("Loading model..."):
                # Load Model Kualitas Tanaman Selada
                model_quality, accuracy = model.quality_model()

            st.write("Enter the values for prediction")
            # Create two columns
            col5, col6 = st.columns(2)

            with col5:
                temperature_2 = st.number_input(
                    "Temperature", format="%.2f", value=25.9, step=0.01
                )
                humidity_2 = st.number_input("Humidity", value=84, step=1)
                light_2 = st.number_input("Light", value=10870, step=1)

            with col6:
                pH_2 = st.number_input("pH", format="%.2f", value=6.6, step=0.01)
                EC_2 = st.number_input("EC", value=983, step=1)
                TDS_2 = st.number_input("TDS", value=493, step=1)
                WaterTemp_2 = st.number_input(
                    "Water Temperature", format="%.2f", value=26.3, step=0.01
                )

            # Create input data for prediction
            input_data = {
                "temperature": temperature_2,
                "humidity": humidity_2,
                "light": light_2,
                "pH": pH_2,
                "EC": EC_2,
                "TDS": TDS_2,
                "WaterTemp": WaterTemp_2,
            }

            # Make prediction
            if st.button("Predict"):
                model.predict_pattern(model_quality, input_data)

            st.markdown(f"#### 📝 Kesimpulan")

            conclusion = cek_optimization.summarize_forecast(df, forecast, periods)
            st.info(f"\n{conclusion}")

            growth_percentage, last_leaf_count, max_forecasted_leaf_count = (
                visualization.calculate_growth_percentage(df, forecast)
            )

            fig = visualization.plot_growth_bar(
                growth_percentage, last_leaf_count, max_forecasted_leaf_count
            )
            st.plotly_chart(fig)

            st.markdown("##### 🔍 Kesimpulan Masing Masing Variabel")

            suggestions = cek_optimization.check_optimization(merged)

            if suggestions:
                for suggestion in suggestions:
                    st.write(suggestion)

            else:
                st.subheader(
                    "✅ Semua variabel berada dalam kondisi optimal untuk pertumbuhan tanaman selada."
                )

            st.markdown("### 🔎 Detail Variabel")

            selected_feature = st.selectbox(
                "🎯 Pilih fitur untuk divisualisasikan:", df.columns[1:]
            )

            visualization.visualize_feature(df, selected_feature)

            st.markdown("#### 🆚 Visualisasi Perbandingan Fitur")

            # Pilih dua fitur yang akan dibandingkan
            feature_a = st.selectbox("Pilih Fitur A", df.columns[1:])
            feature_b = st.selectbox("Pilih Fitur B", df.columns[2:])

            if feature_a and feature_b:
                visualization.visualize_comparison(df, feature_a, feature_b)

        else:
            st.write("Silakan unggah file CSV terlebih dahulu.")

    elif option == "Gunakan contoh file CSV":
        # Use example CSV
        url_example = "https://raw.githubusercontent.com/Vinzzztty/Forecasting-Hidroponik/main/dataset/dummy_data_test.csv"
        df = pd.read_csv(url_example)
        st.write("Menggunakan contoh file CSV dari URL")

        # Convert 'datetime'column to datetime format if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

            if df["datetime"].isnull().any():
                st.error("⚠️ Ada nilai yang tidak bisa dikonversi ke format datetime.")
                return

        # important column
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

        st.markdown("### 📊 Data tanaman yang di Upload")

        st.dataframe(df)

        df_prophet = model.prepare_data(df)

        # Load Model
        models = model.load_model("./model/prophet_model.pkl")

        unique_days = df["datetime"].dt.date.nunique()

        st.info(f"🗓️ Total hari setelah di Tanam: {unique_days} hari")

        with st.spinner(text="⏳ Sedang menganalisis..."):
            time.sleep(2)

        future = models.make_future_dataframe(periods=unique_days, freq="D")

        future["cap"] = 18

        future["hole"] = df_prophet["hole"].iloc[-1]
        future["temperature"] = df_prophet["temperature"].iloc[-1]
        future["humidity"] = df_prophet["humidity"].iloc[-1]
        future["light"] = df_prophet["light"].iloc[-1]
        future["pH"] = df_prophet["pH"].iloc[-1]
        future["EC"] = df_prophet["EC"].iloc[-1]
        future["TDS"] = df_prophet["TDS"].iloc[-1]
        future["WaterTemp"] = df_prophet["WaterTemp"].iloc[-1]

        forecast = models.predict(future)

        forecast_test = forecast[forecast["ds"].isin(df_prophet["ds"])]

        merged = pd.merge(
            df_prophet,
            forecast_test[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            on="ds",
        )

        max_periods = MAX_DAY - unique_days

        periods = st.slider(
            "⏳ Pilih hari untuk Forecasting pertumbuhan daun",
            min_value=unique_days,
            max_value=max_periods,
            step=1,
        )

        future = model.create_future_dataframe(df_prophet, periods=periods)

        future["cap"] = 18

        forecast = model.make_predictions(models, future)

        st.markdown(""" --- """)
        st.markdown(f"### 📈 Hasil Forecasting untuk {periods} Hari Ke Depan")

        st.write(f"🔍 Visualisasi Prediksi")
        fig = visualization.plot_forecast(forecast, periods)

        # st.pyplot(fig)
        st.plotly_chart(fig)

        col1, col2 = st.columns([6, 4])

        with col1:
            max_forecasted_leaf_count = forecast["yhat"].max()

            if max_forecasted_leaf_count <= 6:
                image_path = "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/early_leaf.png?raw=true"
            elif max_forecasted_leaf_count <= 12:
                image_path = "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/mid_leaf.png?raw=true"
            else:
                image_path = "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/high_leaf.png?raw=true"

            # Center the image using HTML and display it
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="{image_path}" alt="Prediksi Jumlah Daun Selada" style="width: 50%;">
                    <p>Prediksi Jumlah Daun Selada</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.write(f"📋 Tabel Prediksi")
            st.dataframe(
                forecast[
                    [
                        "ds",
                        "yhat",
                        "yhat_lower",
                        "yhat_upper",
                    ]
                ]
            )

        warnings.filterwarnings("ignore")

        st.markdown(f"#### Kualitas Tanaman Selada")

        # Display loading spinner while the model is being loaded
        with st.spinner("Loading model..."):
            # Load Model Kualitas Tanaman Selada
            model_quality, accuracy = model.quality_model()

        st.write("Enter the values for prediction")
        # Create two columns
        col5, col6 = st.columns(2)

        with col5:
            temperature_2 = st.number_input(
                "Temperature", format="%.2f", value=25.9, step=0.01
            )
            humidity_2 = st.number_input("Humidity", value=84, step=1)
            light_2 = st.number_input("Light", value=10870, step=1)

        with col6:
            pH_2 = st.number_input("pH", format="%.2f", value=6.6, step=0.01)
            EC_2 = st.number_input("EC", value=983, step=1)
            TDS_2 = st.number_input("TDS", value=493, step=1)
            WaterTemp_2 = st.number_input(
                "Water Temperature", format="%.2f", value=26.3, step=0.01
            )

        # Create input data for prediction
        input_data = {
            "temperature": temperature_2,
            "humidity": humidity_2,
            "light": light_2,
            "pH": pH_2,
            "EC": EC_2,
            "TDS": TDS_2,
            "WaterTemp": WaterTemp_2,
        }

        # Make prediction
        if st.button("Predict"):
            model.predict_pattern(model_quality, input_data)

        st.markdown(f"#### 📝 Kesimpulan")
        conclusion = cek_optimization.summarize_forecast(df, forecast, periods)
        st.info(f"\n{conclusion}")

        growth_percentage, last_leaf_count, max_forecasted_leaf_count = (
            visualization.calculate_growth_percentage(df, forecast)
        )

        fig = visualization.plot_growth_bar(
            growth_percentage, last_leaf_count, max_forecasted_leaf_count, days=periods
        )
        st.plotly_chart(fig)

        st.markdown("##### 🔍 Kesimpulan Masing Masing Variabel")

        suggestions = cek_optimization.check_optimization(merged)

        if suggestions:
            for suggestion in suggestions:
                st.write(suggestion)

        else:
            st.subheader(
                "✅ Semua variabel berada dalam kondisi optimal untuk pertumbuhan tanaman selada."
            )

        st.markdown("### 🔎 Detail Variabel")

        # display visualization

        selected_feature = st.selectbox(
            "🎯 Pilih fitur untuk divisualisasikan:", df.columns[1:]
        )

        visualization.visualize_feature(df, selected_feature)

        st.markdown("#### 🆚 Visualisasi Perbandingan Fitur")

        # Pilih dua fitur yang akan dibandingkan
        feature_a = st.selectbox("Pilih Fitur A", df.columns[1:])
        feature_b = st.selectbox("Pilih Fitur B", df.columns[2:])

        if feature_a and feature_b:
            visualization.visualize_comparison(df, feature_a, feature_b)


if __name__ == "__main__":
    main()
