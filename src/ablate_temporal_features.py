
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "test4_subset.csv"
TEMPORAL_PATH = ROOT / "data" / "test4_temporal_features.csv"
OUTPUT_PATH = (
    ROOT / "data" / "benchmark" / "temporal_experiment"
    / "feature_ablation.csv"
)

BASE = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1"]

OPERATING_HISTORY = [
    f"{col}_lag1" for col in BASE
] + [
    f"{col}_lag5" for col in BASE
] + [
    "current_change_5rows",
    "temperature_change_5rows",
    "current_mean_5rows",
]

VOLTAGE_HISTORY = [
    "CV-mean_lag1",
    "CV-mean_lag5",
    "voltage_change_5rows",
    "voltage_mean_5rows",
]

FEATURE_SETS = {
    "Base only": BASE,
    "Operating history": BASE + OPERATING_HISTORY,
    "Full temporal": BASE + OPERATING_HISTORY + VOLTAGE_HISTORY,
}


def make_model():
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        loss="squared_error",
        random_state=42,
    )


def main():
    raw = pd.read_csv(RAW_PATH)
    temporal = pd.read_csv(TEMPORAL_PATH)

    # The raw data have no missing required values, so temporal rows
    # should correspond to raw rows 6 onward.
    split = int(len(raw) * 0.8)
    train_end = split - 6

    train = temporal.iloc[:train_end].copy()
    valid = temporal.iloc[train_end:].copy()

    if len(valid) != len(raw.iloc[split:]):
        raise ValueError("Validation row counts do not match.")

    if not np.allclose(
        valid["CV-mean"].to_numpy(),
        raw.iloc[split:]["CV-mean"].to_numpy(),
    ):
        raise ValueError("Validation target alignment failed.")

    results = []

    for name, features in FEATURE_SETS.items():
        model = make_model()
        model.fit(train[features], train["CV-mean"])

        pred = model.predict(valid[features])
        actual = valid["CV-mean"].to_numpy()

        results.append({
            "feature_set": name,
            "feature_count": len(features),
            "MAE_V": mean_absolute_error(actual, pred),
            "RMSE_V": np.sqrt(mean_squared_error(actual, pred)),
            "R2": r2_score(actual, pred),
        })

    result = pd.DataFrame(results).round(6)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Temporal feature ablation: Test4 validation ===")
    print(result.to_string(index=False))
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
