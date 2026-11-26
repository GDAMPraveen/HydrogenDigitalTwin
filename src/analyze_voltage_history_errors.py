from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
PRED_PATH = (
    DATA / "benchmark" / "temporal_experiment" / "ood"
    / "temporal_ood_predictions.csv"
)

FEATURES = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1", "CV-mean"]

raw = pd.read_csv(DATA / "test2_subset.csv")
preds = pd.read_csv(PRED_PATH)
preds = preds.loc[preds["experiment"].eq("Test2")].copy()

# Reconstruct the same historical voltage diagnostics from raw Test2 rows.
raw["raw_row_index"] = np.arange(len(raw))
raw["voltage_lag1"] = raw["CV-mean"].shift(1)
raw["voltage_lag5"] = raw["CV-mean"].shift(5)
raw["voltage_change_5rows"] = (
    raw["CV-mean"].shift(1) - raw["CV-mean"].shift(6)
)
raw["voltage_mean_5rows"] = (
    raw["CV-mean"].shift(1).rolling(5, min_periods=5).mean()
)

df = preds.merge(
    raw[[
        "raw_row_index", "voltage_lag1", "voltage_lag5",
        "voltage_change_5rows", "voltage_mean_5rows"
    ]],
    on="raw_row_index",
    how="left",
    validate="one_to_one",
)

df = df.dropna(subset=[
    "voltage_lag1", "voltage_lag5",
    "voltage_change_5rows", "voltage_mean_5rows",
])

print("=== TEST2 ERROR BY RECENT ABSOLUTE VOLTAGE CHANGE ===")
df["change_group"] = pd.cut(
    df["voltage_change_5rows"].abs(),
    bins=[-np.inf, 0.01, 0.05, 0.15, np.inf],
    labels=["<=0.01 V", "0.01–0.05 V", "0.05–0.15 V", ">0.15 V"],
)

print(
    df.groupby("change_group", observed=False).agg(
        rows=("absolute_error_V", "size"),
        MAE=("absolute_error_V", "mean"),
        median_absolute_error=("absolute_error_V", "median"),
        mean_signed_error=("error_V", "mean"),
    ).round(4).to_string()
)

print("\n=== TEST2 LOW-CURRENT ERROR BY RECENT VOLTAGE CHANGE ===")
low = df.loc[df["current_A"] < 0.5].copy()
print(
    low.groupby("change_group", observed=False).agg(
        rows=("absolute_error_V", "size"),
        MAE=("absolute_error_V", "mean"),
        median_absolute_error=("absolute_error_V", "median"),
        mean_signed_error=("error_V", "mean"),
    ).round(4).to_string()
)

print("\n=== TEST2 LOW-CURRENT ERROR BY PREVIOUS VOLTAGE ===")
low["previous_voltage_group"] = pd.cut(
    low["voltage_lag1"],
    bins=[-np.inf, 0.1, 0.4, 0.7, 1.0, 1.3, np.inf],
    labels=["<0.1 V", "0.1–0.4 V", "0.4–0.7 V",
            "0.7–1.0 V", "1.0–1.3 V", ">=1.3 V"],
)

print(
    low.groupby("previous_voltage_group", observed=False).agg(
        rows=("absolute_error_V", "size"),
        MAE=("absolute_error_V", "mean"),
        median_absolute_error=("absolute_error_V", "median"),
        mean_signed_error=("error_V", "mean"),
    ).round(4).to_string()
)

print(
    "\nSigned error = measured voltage - predicted voltage. "
    "Negative means overprediction."
)
