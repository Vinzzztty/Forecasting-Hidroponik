import pandas as pd


def check_optimization(df):
    # Calculate the mean of each feature
    means = df.mean().round(2)

    # Define optimal ranges for each feature
    optimal_conditions = {
        "temperature_x": (25, 28),
        "humidity_x": (50, 70),
        "light_x": (1000, 4000),
        "pH_x": (6.0, 7.0),
        "EC_x": (1200, 1800),
        "TDS_x": (560, 840),
        "WaterTemp_x": (25, 28),
    }

    # Determine if each feature is within optimal range
    def check_optimal(feature, value):
        if feature in optimal_conditions:
            lower, upper = optimal_conditions[feature]
            return lower <= value <= upper
        return True  # Always optimal if no specific range

    # Generate conclusions for each feature
    conclusions = []
    for feature, mean_value in means.items():
        is_optimal = check_optimal(feature, mean_value)

        if is_optimal:
            # Positive statement if the feature is within optimal range
            conclusions.append(
                f"✔️ Rata-rata {feature} dalam kondisi ideal pada nilai {mean_value}. Kondisi ini mendukung pertumbuhan optimal 🌱."
            )
        else:
            # Normative statement without suggesting specific ranges
            conclusions.append(
                f"⚠️ Rata-rata {feature} tercatat pada {mean_value}. Memerlukan perhatian untuk mencapai kondisi yang lebih mendukung."
            )

    return conclusions


def summarize_forecast(df, forecast, periods):
    # Nilai LeafCount terakhir pada data input
    last_leaf_count = df["LeafCount"].iloc[-1]

    # Nilai tertinggi dari hasil forecasting
    max_forecasted_leaf_count = forecast["yhat"].max()

    # Hitung persentase peningkatan
    growth_percentage = (
        (max_forecasted_leaf_count - last_leaf_count) / last_leaf_count
    ) * 100

    conclusion = (
        f"🌿 **Prediksi Pertumbuhan Daun Selada** 🌿\n\n"
        f"📈 Berdasarkan simulasi pertumbuhan daun selada, diperkirakan terjadi peningkatan sebesar "
        f"**{growth_percentage:.2f}%** dari jumlah daun awal 🌱.\n\n"
        f"📅 Pada hari ke-**{periods}**, banyaknya daun diprediksi akan mencapai **{max_forecasted_leaf_count:.0f}** daun 🥬.\n\n"
        f"✨ Tetap jaga kondisi lingkungan agar prediksi pertumbuhan ini dapat tercapai! 💧☀️"
    )

    return conclusion
