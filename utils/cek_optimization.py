import pandas as pd


def check_optimization(df):
    # Menghitung rata rata masing masing features
    means = df.mean().round(2)

    # Membuat aturan untuk menentukan rata rata sudah optimal
    optimal_conditions = {
        "temperature": (25, 28),
        "humidity": (50, 70),
        "light": (1000, 4000),
        "pH": (6.0, 7.0),
        "EC": (1200, 1800),
        "TDS": (560, 840),
        "WaterTemp": (25, 28),
    }

    # cek optimal
    def check_optimal(feature, value):
        if feature in optimal_conditions:
            lower, upper = optimal_conditions[feature]
            return lower <= value <= upper
        return True

    # Membuat kesimpulan
    suggestions = []
    for feature, mean_value in means.items():
        is_optimal = check_optimal(feature, mean_value)
        if not is_optimal:
            lower, upper = optimal_conditions.get(feature, (None, None))
            suggestions.append(
                f"Rata-rata variabel {feature} belum optimal ({round(mean_value, 2)}). "
                f"Harus di antara {lower} dan {upper}."
            )

    return suggestions


def summarize_forecast(df, forecast):
    # Nilai LeafCount terakhir pada data input
    last_leaf_count = df["LeafCount"].iloc[-1]

    # Nilai tertinggi dari hasil forecasting
    max_forecasted_leaf_count = forecast["yhat"].max()

    # Hitung persentase peningkatan
    growth_percentage = (
        (max_forecasted_leaf_count - last_leaf_count) / last_leaf_count
    ) * 100

    conclusion = (
        f"Sesuai dengan data input dan prediksi dalam simulasi pertumbuhan daun tanaman selada, "
        f"pertumbuhan meningkat sebesar {growth_percentage:.2f}% dari hari setelah ditanam. "
        f"Banyak daun pada hari ke-40 diprediksi sebanyak {max_forecasted_leaf_count:.0f}."
    )

    return conclusion
