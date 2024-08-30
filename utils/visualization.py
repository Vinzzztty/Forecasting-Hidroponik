import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import pandas as pd
import plotly.graph_objs as go


# def plot_forecast(model, forecast, periods):
#     fig, ax = plt.subplots(
#         figsize=(10, 5)
#     )  # Increased figure size for better readability

#     # Use Prophet's plot method
#     model.plot(forecast, ax=ax)

#     # Customize the plot
#     ax.set_title(
#         f"Perkiraan Jumlah Daun untuk {periods} Hari ke Depan",
#         fontsize=18,
#         fontweight="bold",
#     )
#     ax.set_xlabel("Tanggal", fontsize=12)
#     ax.set_ylabel("Jumlah Daun", fontsize=12)

#     # Improve the appearance of x-ticks
#     plt.xticks(rotation=45, fontsize=12)
#     plt.yticks(fontsize=12)
#     ax.grid(
#         True, linestyle="--", alpha=0.7
#     )  # Make the grid lines dashed and semi-transparent

#     # Customize the legend
#     legend = ax.get_legend()
#     if legend:
#         legend.set_frame_on(False)  # Turn off legend frame for better appearance
#         legend.set_fontsize(12)  # Adjust legend font size

#     # Add annotations or markers if needed
#     # Example: Highlight a specific forecast point
#     max_y = forecast["yhat"].max()
#     max_date = forecast.loc[forecast["yhat"].idxmax(), "ds"]
#     ax.annotate(
#         "Puncak Perkiraan",
#         xy=(max_date, max_y),
#         xytext=(max_date, max_y * 1.1),
#         arrowprops=dict(facecolor="red", shrink=0.05),
#         fontsize=12,
#         color="red",
#         ha="center",
#     )

#     # Adjust layout
#     plt.tight_layout()

#     return fig


def plot_forecast(forecast, periods):
    # Create a figure
    fig = go.Figure()

    # Add the forecasted values
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="red", dash="dash"),
        )
    )

    # Add the uncertainty intervals
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"].tolist() + forecast["ds"][::-1].tolist(),
            y=forecast["yhat_upper"].tolist() + forecast["yhat_lower"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            showlegend=False,
            name="Uncertainty Interval",
        )
    )

    # Highlight the maximum forecast point
    max_y = forecast["yhat"].max()
    max_date = forecast.loc[forecast["yhat"].idxmax(), "ds"]
    fig.add_trace(
        go.Scatter(
            x=[max_date],
            y=[max_y],
            mode="markers+text",
            name="Puncak Perkiraan",
            text=["Puncak Perkiraan"],
            textposition="top center",
            marker=dict(color="red", size=10),
        )
    )

    # Customize the layout
    fig.update_layout(
        title=f"Perkiraan Jumlah Daun untuk {periods} Hari ke Depan",
        xaxis_title="Tanggal",
        yaxis_title="Jumlah Daun",
        legend=dict(font=dict(size=12)),
        xaxis=dict(tickformat="%Y-%m-%d"),
        hovermode="x unified",
    )

    return fig


def calculate_growth_percentage(df, forecast):
    # Last actual leaf count from the input data
    last_leaf_count = df["LeafCount"].iloc[-1]

    # Max forecasted leaf count
    max_forecasted_leaf_count = forecast["yhat"].max()

    # Calculate percentage increase
    growth_percentage = (
        (max_forecasted_leaf_count - last_leaf_count) / last_leaf_count
    ) * 100

    return growth_percentage, last_leaf_count, max_forecasted_leaf_count


def plot_growth_bar(
    growth_percentage, last_leaf_count, max_forecasted_leaf_count, days=40
):
    fig = go.Figure()

    # Add bars for initial and forecasted leaf count
    fig.add_trace(
        go.Bar(
            x=["Hari Terakhir", f"Hari ke-{days} (Forecast)"],
            y=[last_leaf_count, max_forecasted_leaf_count],
            text=[
                f"{last_leaf_count:.0f} Daun",
                f"{max_forecasted_leaf_count:.0f} Daun (+{growth_percentage:.2f}%)",
            ],
            textposition="auto",
            marker=dict(color=["blue", "red"]),
            name="Jumlah Daun",
        )
    )

    # Customize layout
    fig.update_layout(
        title=f"Kenaikan Persentase Jumlah Daun Selama {days} Hari ke Depan",
        xaxis_title="Hari",
        yaxis_title="Jumlah Daun",
        template="plotly_white",
        yaxis=dict(
            range=[0, max(max_forecasted_leaf_count * 1.2, last_leaf_count * 1.2)]
        ),
    )

    return fig
