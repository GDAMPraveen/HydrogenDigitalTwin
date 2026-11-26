
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    ROOT / "data" / "benchmark" / "temporal_experiment"
    / "ood" / "temporal_ood_predictions.csv"
)
OUTPUT_DIR = ROOT / "data" / "benchmark" / "temporal_experiment" / "ood"


def summarize_errors(df, group_columns):
    return (
        df.groupby(group_columns, observed=True)["absolute_error_V"]
        .agg(
            count="count",
            MAE="mean",
            median_AE="median",
            p90_AE=lambda x: x.quantile(0.90),
            p95_AE=lambda x: x.quantile(0.95),
            max_AE="max",
        )
        .round(5)
    )


def main():
    df = pd.read_csv(INPUT_PATH)

    required = {
        "experiment",
        "current_A",
        "temperature_C",
        "absolute_error_V",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["current_band"] = pd.cut(
        df["current_A"],
        bins=[-float("inf"), 0.5, 3, float("inf")],
        labels=["low (<0.5 A)", "mid (0.5-3 A)", "high (>=3 A)"],
    )
    df["temperature_band"] = pd.cut(
        df["temperature_C"],
        bins=[-float("inf"), 30, 50, 65, 75, float("inf")],
        labels=["<30 C", "30-50 C", "50-65 C", "65-75 C", ">75 C"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall = summarize_errors(df, ["experiment"])
    by_current = summarize_errors(df, ["experiment", "current_band"])
    by_temperature = summarize_errors(df, ["experiment", "temperature_band"])
    by_both = summarize_errors(
        df, ["experiment", "temperature_band", "current_band"]
    )

    overall.to_csv(OUTPUT_DIR / "risk_overall.csv")
    by_current.to_csv(OUTPUT_DIR / "risk_by_current.csv")
    by_temperature.to_csv(OUTPUT_DIR / "risk_by_temperature.csv")
    by_both.to_csv(OUTPUT_DIR / "risk_by_current_temperature.csv")

    print("\n=== Overall error by experiment ===")
    print(overall.to_string())

    print("\n=== Error by experiment and current band ===")
    print(by_current.to_string())

    print("\n=== Error by experiment and temperature band ===")
    print(by_temperature.to_string())

    print("\n=== Error by experiment, temperature, and current ===")
    print(by_both.to_string())

    print("\nSaved risk summaries to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
