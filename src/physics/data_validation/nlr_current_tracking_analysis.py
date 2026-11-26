from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT / "data" / "nlr" / "raw" /
    "characterization" / "characterization_200.zip"
)

OUTPUT_DIR = ROOT / "reports" / "physics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = "Unnamed: 0"
ACTUAL_CURRENT = "H2E_f_PSU_A_Current"
COMMAND_CURRENT = "H2E_f_NLR_CurrentCmd (A)"

# Load original characterization data
with ZipFile(DATA_PATH) as archive:
    with archive.open("characterization_200.csv") as file:
        df = pd.read_csv(BytesIO(file.read()))

# Validate required columns
required = [TIMESTAMP, ACTUAL_CURRENT, COMMAND_CURRENT]
missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df[required].isna().any().any():
    raise ValueError("Missing values found in current-tracking inputs.")

# Timestamp and tracking calculations
df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
df["elapsed_s"] = (
    df[TIMESTAMP] - df[TIMESTAMP].iloc[0]
).dt.total_seconds()

df["tracking_error_A"] = (
    df[ACTUAL_CURRENT] - df[COMMAND_CURRENT]
)

df["absolute_tracking_error_A"] = (
    df["tracking_error_A"].abs()
)

df["command_change_A"] = df[COMMAND_CURRENT].diff()

# Period-level performance
period_edges = [-1, 120, 600, 1800, 3600, float("inf")]
period_labels = [
    "0-120s",
    "121-600s",
    "601-1800s",
    "1801-3600s",
    ">3600s",
]

df["period"] = pd.cut(
    df["elapsed_s"],
    bins=period_edges,
    labels=period_labels,
)

period_summary = (
    df.groupby("period", observed=True)
    .agg(
        samples=("tracking_error_A", "size"),
        mean_error_A=("tracking_error_A", "mean"),
        MAE_A=("absolute_tracking_error_A", "mean"),
        error_std_A=("tracking_error_A", "std"),
        maximum_absolute_error_A=(
            "absolute_tracking_error_A", "max"
        ),
    )
    .reset_index()
)

# Command transition analysis
transition_indices = df.index[
    df["command_change_A"].notna()
    & df["command_change_A"].ne(0)
]

transition_rows = []

for index in transition_indices:
    row = df.loc[index]
    next_row = df.loc[index + 1] if index + 1 < len(df) else None

    transition_rows.append({
        "timestamp": row[TIMESTAMP],
        "elapsed_s": row["elapsed_s"],
        "previous_command_A": df.loc[index - 1, COMMAND_CURRENT],
        "new_command_A": row[COMMAND_CURRENT],
        "command_change_A": row["command_change_A"],
        "actual_current_at_transition_A": row[ACTUAL_CURRENT],
        "error_at_transition_A": row["tracking_error_A"],
        "actual_current_next_sample_A": (
            next_row[ACTUAL_CURRENT] if next_row is not None else None
        ),
        "error_next_sample_A": (
            next_row["tracking_error_A"] if next_row is not None else None
        ),
    })

transitions = pd.DataFrame(transition_rows)

# Overall validation summary
summary = pd.DataFrame([
    {"metric": "total_samples", "value": len(df)},
    {
        "metric": "overall_mean_tracking_error_A",
        "value": df["tracking_error_A"].mean(),
    },
    {
        "metric": "overall_MAE_A",
        "value": df["absolute_tracking_error_A"].mean(),
    },
    {
        "metric": "overall_max_absolute_error_A",
        "value": df["absolute_tracking_error_A"].max(),
    },
    {
        "metric": "command_transition_count",
        "value": len(transition_indices),
    },
    {
        "metric": "upward_transition_count",
        "value": int(df["command_change_A"].gt(0).sum()),
    },
    {
        "metric": "downward_transition_count",
        "value": int(df["command_change_A"].lt(0).sum()),
    },
])

# Save reports
period_summary.to_csv(
    OUTPUT_DIR / "nlr_current_tracking_period_summary.csv",
    index=False,
)

transitions.to_csv(
    OUTPUT_DIR / "nlr_current_tracking_transitions.csv",
    index=False,
)

summary.to_csv(
    OUTPUT_DIR / "nlr_current_tracking_validation_summary.csv",
    index=False,
)

print("NLR current-tracking analysis completed.")
print("\nPeriod performance:")
print(period_summary.to_string(index=False))

print("\nCommand transitions:")
print(transitions.to_string(index=False))

print("\nOverall validation summary:")
print(summary.to_string(index=False))

print(f"\nReports saved to: {OUTPUT_DIR}")
