import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def set_page_config():
    """Set the initial page configuration."""
    st.set_page_config(
        page_icon="https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/logo_hijau.png?raw=true",
        page_title="Hydrosim - Home",
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
        <img src='https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/new_banner_800.png?raw=true' class='header-image'/>
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

    st.title("🌿 Welcome to the Home Page")
    st.header("🎯 Tujuan")
    st.write(
        """
    HydroSim bertujuan untuk **meramalkan pertumbuhan tanaman hidroponik** dengan menggunakan data historis yang meliputi jumlah daun serta variabel lingkungan seperti suhu, kelembapan, cahaya, pH, dan lainnya. Dengan peramalan ini, pengguna dapat mengoptimalkan kondisi pertumbuhan dan meningkatkan hasil panen.
    """
    )

    st.header("✨ Manfaat")
    st.write(
        """
    - **Prediksi Pertumbuhan**: Memberikan gambaran mengenai perkembangan tanaman hidroponik di masa depan berdasarkan data historis.
    - **Optimalisasi Kondisi**: Membantu dalam menyesuaikan variabel lingkungan untuk mencapai pertumbuhan tanaman yang maksimal.
    - **Pengambilan Keputusan yang Lebih Baik**: Memungkinkan pengguna untuk membuat keputusan yang lebih tepat dalam mengelola kebun hidroponik.
    - **Efisiensi Waktu dan Sumber Daya**: Mengurangi risiko kegagalan tanam dan menghemat sumber daya melalui pemantauan yang lebih efektif.
    """
    )

    # Set up the two columns layout with different widths
    st.header("🔍 Kesimpulan")
    # col1, col2 = st.columns([3, 7])  # 70% and 30%

    # Content for the first column
    # with col1:
    st.markdown(
        "Model forecasting menggunakan algoritma Prophet menghasilkan metrik evaluasi sebagai berikut:"
    )

    st.markdown("- **RMSE (Root Mean Square Error)**: 1.82")
    st.markdown("- **MAE (Mean Absolute Error)**: 1.49")

    st.markdown(
        "Hasil menunjukkan bahwa model memiliki akurasi yang baik dengan kesalahan prediksi yang relatif rendah."
    )

    # Content for the second column
    # with col2:
    st.image(
        "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/evaluasi_model.png?raw=true",
        caption="Evaluasi Model",
    )

    # Set up the two columns layout with different widths
    st.header("🔍 Perbandingan Model ARIMA dan Prophet")

    # Data evaluasi model
    data = {
        "Model": ["ARIMA", "Prophet"],
        "RMSE": [8.970086115676192, 1.8154298562350784],
        "MAE": [8.170746410714356, 1.4857125402032283],
    }

    # Membuat DataFrame
    df = pd.DataFrame(data)

    col1, col2 = st.columns([3, 7])

    with col1:
        st.write("### Tabel Evaluasi Model")
        st.dataframe(df)

    with col2:
        st.image(
            "https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/perbandingan_model.png?raw=true",
            caption="Evaluasi Model",
        )

    # Kesimpulan
    st.write("### Kesimpulan")
    st.markdown(
        """
        Hasil evaluasi menunjukkan bahwa model Prophet mengungguli model ARIMA dalam metrik RMSE dan MAE.
        Prophet memiliki RMSE (1.815) dan MAE (1.486) yang lebih rendah dibandingkan dengan ARIMA, yang memiliki RMSE (8.970) dan MAE (8.171).
        Hal ini menunjukkan bahwa model Prophet memberikan prediksi yang lebih akurat.
        """
    )


if __name__ == "__main__":
    main()
