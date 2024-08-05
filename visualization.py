import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import numpy as np

# Define custom color themes
REAL_DATA_COLOR = "blue"
FORECAST_COLOR = "orange"


def plot_forecast(model, forecast, periods):
    fig, ax = plt.subplots(
        figsize=(14, 7)
    )  # Increased figure size for better readability

    # Use Prophet's plot method
    model.plot(forecast, ax=ax)

    # Customize the plot
    ax.set_title(
        f"Perkiraan Jumlah Daun untuk {periods} Hari ke Depan",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_xlabel("Tanggal", fontsize=16)
    ax.set_ylabel("Jumlah Daun", fontsize=16)

    # Improve the appearance of x-ticks
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    ax.grid(
        True, linestyle="--", alpha=0.7
    )  # Make the grid lines dashed and semi-transparent

    # Customize the legend
    legend = ax.get_legend()
    if legend:
        legend.set_frame_on(False)  # Turn off legend frame for better appearance
        legend.set_fontsize(12)  # Adjust legend font size

    # Add annotations or markers if needed
    # Example: Highlight a specific forecast point
    max_y = forecast["yhat"].max()
    max_date = forecast.loc[forecast["yhat"].idxmax(), "ds"]
    ax.annotate(
        "Puncak Perkiraan",
        xy=(max_date, max_y),
        xytext=(max_date, max_y * 1.1),
        arrowprops=dict(facecolor="red", shrink=0.05),
        fontsize=12,
        color="red",
        ha="center",
    )

    # Adjust layout
    plt.tight_layout()

    return fig


def display_variable_info(df, forecast):
    st.write("## Informasi Variabel")

    if "ds" not in df.columns:
        st.write(
            "Dataframe tidak memiliki kolom 'ds'. Pastikan data sudah dimuat dengan benar."
        )
        return

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

        # Plot historical data with real data color theme
        fig, ax = plt.subplots(
            figsize=(12, 6)
        )  # Adjust figure size for better readability
        ax.plot(df["ds"], df[col], color=REAL_DATA_COLOR, label=col)
        ax.set_xlabel("Tanggal")
        ax.set_ylabel(col)
        ax.legend(loc="upper right")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Set x-axis major locator to a fixed interval (e.g., every 10 days)
        locator = mdates.DayLocator(interval=max(1, len(df) // 10))
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        st.pyplot(fig)

        # Display statistics
        st.write("#### Statistik")
        st.write(f"- Nilai Tertinggi: {np.max(df[col]):.2f}")
        st.write(f"- Nilai Terendah: {np.min(df[col]):.2f}")
        st.write(f"- Rata-rata: {np.mean(df[col]):.2f}")

        # Plot forecasted values with forecast color theme if available
        if "ds" in forecast.columns:
            st.write("#### Pola Pergerakan selama Ramalan")
            fig, ax = plt.subplots(
                figsize=(12, 6)
            )  # Adjust figure size for better readability
            ax.plot(
                forecast["ds"],
                forecast[col],
                color=FORECAST_COLOR,
                label="Ramalan " + col,
            )
            ax.set_xlabel("Tanggal")
            ax.set_ylabel(col)
            ax.legend(loc="upper right")

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)

            # Set x-axis major locator to a fixed interval (e.g., every 10 days)
            locator = mdates.DayLocator(interval=max(1, len(forecast) // 10))
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

            st.pyplot(fig)
