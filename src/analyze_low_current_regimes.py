
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "data" / "benchmark" / "temporal_experiment"
    / "ood" / "temporal_ood_predictions.csv"
)

df = pd.read_csv(INPUT_PATH)

# Focus on low-current operating conditions.
df = df[df["current_A"] < 0.5].copy()

# Group observations by the measured voltage regime.
df["measured_voltage_band"] = pd.cut(
    df["measured_voltage_V"],
    bins=[
        -float("inf"),
        0.1,
        0.4,
        0.7,
        1.0,
        1.3,
        float("inf"),
    ],
    labels=[
        "<0.1 V",
        "0.1-0.4 V",
        "0.4-0.7 V",
        "0.7-1.0 V",
        "1.0-1.3 V",
        ">=1.3 V",
    ],
)

summary = (
    df.groupby(
        ["experiment", "measured_voltage_band"],
        observed=True,
    )
    .agg(
        count=("error_V", "count"),
        measured_mean_V=("measured_voltage_V", "mean"),
        predicted_mean_V=("predicted_voltage_V", "mean"),
        signed_bias_V=("error_V", "mean"),
        MAE_V=("absolute_error_V", "mean"),
        median_AE_V=("absolute_error_V", "median"),
    )
    .round(5)
)

print("\n=== Low-current errors by measured voltage regime ===")
print(summary.to_string())

print("\nInterpretation:")
print("Positive signed bias = model underpredicts voltage.")
print("Negative signed bias = model overpredicts voltage.")

OUTPUT_PATH = (
    ROOT / "data" / "benchmark" / "temporal_experiment"
    / "ood" / "low_current_regime_analysis.csv"
)

summary.to_csv(OUTPUT_PATH)
print("\nSaved:", OUTPUT_PATH)
