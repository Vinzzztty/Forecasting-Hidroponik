import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import numpy as np


def plot_forecast(model, forecast, periods):
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )  # Increased figure size for better readability

    # Use Prophet's plot method
    model.plot(forecast, ax=ax)

    # Customize the plot
    ax.set_title(
        f"Perkiraan Jumlah Daun untuk {periods} Hari ke Depan",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_xlabel("Tanggal", fontsize=12)
    ax.set_ylabel("Jumlah Daun", fontsize=12)

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
