
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "data" / "benchmark" / "temporal_experiment"
    / "ood" / "temporal_ood_predictions.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "benchmark" / "temporal_experiment"
    / "ood" / "low_current_timeline.csv"
)

df = pd.read_csv(INPUT_PATH)

# Diagnose Test2 only; keep the independent test data unchanged.
df = df[
    (df["experiment"] == "Test2")
    & (df["current_A"] < 0.5)
].copy()

# Row index is a position in the CSV, not a verified timestamp.
df["row_block"] = (df["raw_row_index"] // 1000) * 1000

summary = (
    df.groupby("row_block", observed=True)
    .agg(
        count=("error_V", "count"),
        mean_current_A=("current_A", "mean"),
        mean_temperature_C=("temperature_C", "mean"),
        mean_measured_voltage_V=("measured_voltage_V", "mean"),
        mean_predicted_voltage_V=("predicted_voltage_V", "mean"),
        signed_bias_V=("error_V", "mean"),
        MAE_V=("absolute_error_V", "mean"),
        median_AE_V=("absolute_error_V", "median"),
    )
    .round(5)
)

print("\n=== Test2 low-current errors by 1000-row block ===")
print(summary.to_string())

summary.to_csv(OUTPUT_PATH)
print("\nSaved:", OUTPUT_PATH)
