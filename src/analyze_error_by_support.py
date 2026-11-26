import pandas as pd

MATCH_FILE = "data/low_current_nearest_matches.csv"
PRED_FILE = "data/benchmark/temporal_experiment/ood/temporal_ood_predictions.csv"

REFERENCE_DISTANCE = 0.0840

matches = pd.read_csv(MATCH_FILE)
preds = pd.read_csv(PRED_FILE)

# Keep Test2 low-current rows and join their existing OOD predictions.
df = matches.merge(
    preds.loc[preds["experiment"].eq("Test2")],
    left_on="test2_row",
    right_on="raw_row_index",
    how="inner",
    suffixes=("_match", "_pred"),
)

df["support_group"] = df["scaled_distance"].le(
    REFERENCE_DISTANCE
).map({
    True: "At/below Test4 P95 distance",
    False: "Above Test4 P95 distance",
})

print("=== TEST2 LOW-CURRENT ERROR BY SUPPORT GROUP ===")
summary = df.groupby("support_group").agg(
    rows=("absolute_error_V", "size"),
    median_distance=("scaled_distance", "median"),
    MAE=("absolute_error_V", "mean"),
    median_absolute_error=("absolute_error_V", "median"),
    p90_absolute_error=("absolute_error_V", lambda x: x.quantile(0.90)),
    mean_signed_error=("error_V", "mean"),
)
print(summary.round(4).to_string())

print("\n=== SAME ANALYSIS FOR LOW-VOLTAGE TEST2 ROWS (CV-mean < 0.7 V) ===")
low_voltage = df.loc[df["measured_voltage_V"] < 0.7]
summary_low = low_voltage.groupby("support_group").agg(
    rows=("absolute_error_V", "size"),
    median_distance=("scaled_distance", "median"),
    MAE=("absolute_error_V", "mean"),
    median_absolute_error=("absolute_error_V", "median"),
    mean_signed_error=("error_V", "mean"),
)
print(summary_low.round(4).to_string())

print(
    "\nSigned error = measured voltage - predicted voltage. "
    "Negative means overprediction."
)
