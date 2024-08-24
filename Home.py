import streamlit as st


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
        <img src='https://github.com/Vinzzztty/Forecasting-Hidroponik/blob/V2/assets/banner_800.png?raw=true' class='header-image'/>
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

    st.title("Welcome to the Home Page")
    st.header("Tujuan")
    st.write(
        """
    HydroSim bertujuan untuk **meramalkan pertumbuhan tanaman hidroponik** dengan menggunakan data historis yang meliputi jumlah daun serta variabel lingkungan seperti suhu, kelembapan, cahaya, pH, dan lainnya. Dengan peramalan ini, pengguna dapat mengoptimalkan kondisi pertumbuhan dan meningkatkan hasil panen.
    """
    )

    st.header("Manfaat")
    st.write(
        """
    - **Prediksi Pertumbuhan**: Memberikan gambaran mengenai perkembangan tanaman hidroponik di masa depan berdasarkan data historis.
    - **Optimalisasi Kondisi**: Membantu dalam menyesuaikan variabel lingkungan untuk mencapai pertumbuhan tanaman yang maksimal.
    - **Pengambilan Keputusan yang Lebih Baik**: Memungkinkan pengguna untuk membuat keputusan yang lebih tepat dalam mengelola kebun hidroponik.
    - **Efisiensi Waktu dan Sumber Daya**: Mengurangi risiko kegagalan tanam dan menghemat sumber daya melalui pemantauan yang lebih efektif.
    """
    )


if __name__ == "__main__":
    main()
