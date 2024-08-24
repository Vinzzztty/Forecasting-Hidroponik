import streamlit as st


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

    st.title("Welcome to How to Works Page")

    st.header("📊 Petunjuk Sebelum Memasukkan File CSV:")

    st.markdown(
        """
        - **Pastikan file CSV Anda memiliki kolom yang diperlukan**:
          - `datetime`
          - `LeafCount`
          - `hole`
          - `temperature`
          - `humidity`
          - `light`
          - `pH`
          - `EC`
          - `TDS`
          - `WaterTemp`
        """
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

    st.image(
        "https://example.com/path/to/your/csv-sample.png",
        caption="Contoh Format CSV",
        use_column_width=True,
    )

    st.header("🛠️ Langkah-Langkah Penggunaan:")
    st.write("Ikuti langkah-langkah berikut untuk memulai:")
    st.markdown("1. **Unggah file CSV** yang sesuai dengan petunjuk di atas.")
    st.markdown(
        "2. **Tinjau data yang diunggah** untuk memastikan bahwa semua kolom telah terbaca dengan benar."
    )
    st.markdown("3. **Pilih parameter** yang ingin Anda analisis atau ramalkan.")
    st.markdown(
        "4. **Jalankan simulasi** untuk mendapatkan prediksi pertumbuhan tanaman hidroponik Anda."
    )
    st.markdown(
        "5. **Tinjau hasil prediksi** dan sesuaikan variabel lingkungan jika diperlukan."
    )

    st.header("💡 Tips Penggunaan:")
    st.markdown(
        "- **Simpan salinan dataset** sebelum melakukan perubahan, agar Anda memiliki data asli untuk referensi."
    )
    st.markdown(
        "- **Gunakan data terbaru** untuk mendapatkan hasil prediksi yang lebih akurat."
    )
    st.markdown(
        "- Jika Anda menemui masalah dengan format atau kolom yang hilang, **periksa kembali file CSV Anda** dan pastikan sesuai dengan panduan di atas."
    )


if __name__ == "__main__":
    main()
