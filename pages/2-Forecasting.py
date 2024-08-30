import streamlit as st
import pandas as pd
from utils import model, visualization, cek_optimization
from prophet import Prophet
import matplotlib.pyplot as plt
import time

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
            "![Logo](https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/hijau.png?raw=true)"
        )


def main():
    set_page_config()
    inject_custom_css()

    render_sidebar()

    st.title("Welcome to Forecasting Page")

    # df_train_url = "https://raw.githubusercontent.com/Vinzzztty/Forecasting-Hidroponik/main/dataset/dataset_train_final.csv"
    # df_train = pd.read_csv(df_train_url)

    # st.write("### Unggah File CSV")
    # uploaded_file = st.file_uploader(
    #     "Unggah file CSV untuk dilakukan prediksi", type=["csv"]
    # )

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
                    st.error("Ada nilai yang tidak bisa dikonversi ke format datetime.")
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

            st.markdown("### Data tanaman yang di Upload")

            st.dataframe(df)

            df_prophet = model.prepare_data(df)

            # Load Model
            models = model.load_model("./model/prophet_model.pkl")

            unique_days = df["datetime"].dt.date.nunique()

            st.info(f"Total hari setelah di Tanam: {unique_days} hari")

            with st.spinner(text="Tunggu hasil analisis"):
                time.sleep(5)
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

            periods = MAX_DAY - unique_days

            # Add select box for the height of the tanaman
            height_option = st.selectbox(
                "Pilih berat tanaman (gram):", options=[100, 150]
            )

            # Set cap value based on the selected height option
            new_cap = 18 if height_option == 100 else 23

            # st.write(f"Cap value set to: {new_cap}")

            future = model.create_future_dataframe(df_prophet, periods=periods)

            future["cap"] = new_cap

            forecast = model.make_predictions(models, future)

            st.markdown(""" --- """)
            st.markdown(f"### Hasil Forecasting untuk {periods} Ke-Depan")

            col1, col2 = st.columns([6, 4])

            with col1:
                st.write(f"Visualisasi Prediksi")
                fig = visualization.plot_forecast(forecast, periods)

                # st.pyplot(fig)
                st.plotly_chart(fig)

            with col2:
                st.write(f"Tabel Prediksi")
                st.dataframe(
                    forecast[
                        [
                            "ds",
                            "yhat",
                            "yhat_lower",
                            "yhat_upper",
                            # "hole",
                            # "temperature",
                            # "humidity",
                            # "light",
                            # "pH",
                            # "EC",
                            # "TDS",
                            # "WaterTemp",
                        ]
                    ]
                )
            st.markdown(f"#### Kesimpulan")

            conclusion = cek_optimization.summarize_forecast(df, forecast)
            st.info(f"\n{conclusion}")

            growth_percentage, last_leaf_count, max_forecasted_leaf_count = (
                visualization.calculate_growth_percentage(df, forecast)
            )

            fig = visualization.plot_growth_bar(
                growth_percentage, last_leaf_count, max_forecasted_leaf_count
            )
            st.plotly_chart(fig)

            st.markdown("##### Kesimpulan Masing Masing Features")

            suggestions = cek_optimization.check_optimization(merged)

            if suggestions:
                for suggestion in suggestions:
                    st.write(suggestion)

            else:
                st.subheader(
                    "Semua features berada dalam kondisi optimal untuk pertumbuhan tanaman selada."
                )

        else:
            st.write("Silakan unggah file CSV terlebih dahulu.")

    elif option == "Gunakan contoh file CSV":
        # Use example CSV
        url_example = "https://raw.githubusercontent.com/Vinzzztty/Forecasting-Hidroponik/main/dataset/dummy_data_test.csv"
        df = pd.read_csv(url_example)
        st.write("Menggunakan contoh file CSV dari URL:")

        # st.markdown("### Data tanaman yang di Upload")

        # st.dataframe(df)

        # Convert 'datetime'column to datetime format if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

            if df["datetime"].isnull().any():
                st.error("Ada nilai yang tidak bisa dikonversi ke format datetime.")
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

        st.markdown("### Data tanaman yang di Upload")

        st.dataframe(df)

        df_prophet = model.prepare_data(df)

        # Load Model
        models = model.load_model("./model/prophet_model.pkl")

        unique_days = df["datetime"].dt.date.nunique()

        st.info(f"Total hari setelah di Tanam: {unique_days} hari")

        with st.spinner(text="Tunggu hasil analisis"):
            time.sleep(5)
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

        periods = MAX_DAY - unique_days

        # Add select box for the height of the tanaman
        height_option = st.selectbox("Pilih berat tanaman (gram):", options=[100, 150])

        # Set cap value based on the selected height option
        new_cap = 18 if height_option == 100 else 23

        # st.write(f"Cap value set to: {new_cap}")

        future = model.create_future_dataframe(df_prophet, periods=periods)

        future["cap"] = new_cap

        forecast = model.make_predictions(models, future)

        st.markdown(""" --- """)
        st.markdown(f"### Hasil Forecasting untuk {periods} Ke-Depan")

        col1, col2 = st.columns([6, 4])

        with col1:
            st.write(f"Visualisasi Prediksi")
            fig = visualization.plot_forecast(forecast, periods)

            # st.pyplot(fig)
            st.plotly_chart(fig)

        with col2:
            st.write(f"Tabel Prediksi")
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

        st.markdown(f"#### Kesimpulan")
        conclusion = cek_optimization.summarize_forecast(df, forecast)
        st.info(f"\n{conclusion}")

        growth_percentage, last_leaf_count, max_forecasted_leaf_count = (
            visualization.calculate_growth_percentage(df, forecast)
        )

        fig = visualization.plot_growth_bar(
            growth_percentage, last_leaf_count, max_forecasted_leaf_count
        )
        st.plotly_chart(fig)

        st.markdown("##### Kesimpulan Masing Masing Features")

        suggestions = cek_optimization.check_optimization(merged)

        if suggestions:
            for suggestion in suggestions:
                st.write(suggestion)

        else:
            st.subheader(
                "Semua features berada dalam kondisi optimal untuk pertumbuhan tanaman selada."
            )


if __name__ == "__main__":
    main()
