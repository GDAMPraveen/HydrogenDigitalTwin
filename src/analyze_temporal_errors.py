from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/benchmark/temporal_experiment/validation_predictions.csv")
OUTPUT_DIR = Path("data/benchmark/temporal_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

models = {
    "Baseline": (
        "GradientBoosting_baseline_predicted_V",
        "GradientBoosting_baseline_error_V",
    ),
    "Temporal": (
        "GradientBoosting_temporal_predicted_V",
        "GradientBoosting_temporal_error_V",
    ),
}

def metrics(part, model_name, pred_col, error_col):
    measured = part["measured_voltage_V"]
    predicted = part[pred_col]
    error = part[error_col]  # measured - predicted

    return {
        "model": model_name,
        "rows": len(part),
        "MAE_V": np.mean(np.abs(error)),
        "RMSE_V": np.sqrt(np.mean(error ** 2)),
        "bias_measured_minus_predicted_V": error.mean(),
        "min_prediction_V": predicted.min(),
        "max_prediction_V": predicted.max(),
    }

# Current bands help identify where each model performs poorly.
df["operating_region"] = pd.cut(
    df["current_A"],
    bins=[-np.inf, 0.5, 3.0, np.inf],
    labels=["Low current (<0.5 A)", "Intermediate (0.5–3 A)", "High current (>=3 A)"],
    right=False,
)

rows = []

for region, part in df.groupby("operating_region", observed=True):
    for name, (pred_col, error_col) in models.items():
        result = metrics(part, name, pred_col, error_col)
        result["region"] = str(region)
        rows.append(result)

# Inspect the last 500 validation rows as a shutdown-focused window.
tail = df.tail(500).copy()

for name, (pred_col, error_col) in models.items():
    result = metrics(tail, name, pred_col, error_col)
    result["region"] = "Final 500 validation rows"
    rows.append(result)

# Also calculate actual absolute-error quantiles for the full validation set.
for name, (pred_col, error_col) in models.items():
    absolute_error = df[error_col].abs()
    print(f"\n{name} absolute-error quantiles (V):")
    print(
        absolute_error.quantile([0.50, 0.90, 0.95, 0.99])
        .rename(index={0.50: "50%", 0.90: "90%", 0.95: "95%", 0.99: "99%"})
        .to_string()
    )

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT_DIR / "operating_region_error_metrics.csv", index=False)

print("\nError metrics by operating region:")
print(summary.round(6).to_string(index=False))

print("\nFinal 20 validation rows:")
columns = [
    "raw_row_index",
    "current_A",
    "measured_voltage_V",
    models["Baseline"][0],
    models["Temporal"][0],
]
print(df[columns].tail(20).to_string(index=False))

print(f"\nSaved metrics to: {OUTPUT_DIR / 'operating_region_error_metrics.csv'}")
