
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "test4_subset.csv"
OUTPUT_PATH = ROOT / "data" / "test4_temporal_features.csv"

FEATURES = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1"]
TARGET = "CV-mean"


def main():
    df = pd.read_csv(DATA_PATH)

    # Preserve the original chronological row order.
    data = df[FEATURES + [TARGET]].copy()

    # Historical operating measurements and voltage.
    for col in FEATURES + [TARGET]:
        data[f"{col}_lag1"] = data[col].shift(1)
        data[f"{col}_lag5"] = data[col].shift(5)

    # Changes over the previous five rows, ending at t-1.
    data["current_change_5rows"] = (
        data["PS-I-MON"].shift(1) - data["PS-I-MON"].shift(6)
    )
    data["voltage_change_5rows"] = (
        data[TARGET].shift(1) - data[TARGET].shift(6)
    )
    data["temperature_change_5rows"] = (
        data["T-ELY-CH1"].shift(1) - data["T-ELY-CH1"].shift(6)
    )

    # Historical voltage average: rows t-5 through t-1 only.
    data["voltage_mean_5rows"] = (
        data[TARGET].shift(1).rolling(window=5, min_periods=5).mean()
    )

    # Current operating measurement and recent current average.
    data["current_mean_5rows"] = (
        data["PS-I-MON"].rolling(window=5, min_periods=5).mean()
    )

    data = data.dropna().reset_index(drop=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Rows:", len(data))
    print("Features:", len(data.columns) - 1)
    print("\nLast 5 rows:")
    print(data.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
