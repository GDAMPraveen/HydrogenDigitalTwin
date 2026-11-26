from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = ROOT / "data" / "norce" / "test4_subset.csv"
REPORT_DIR = ROOT / "reports" / "physics" / "norce"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

required = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1", "CV-mean"]
df = df[required].apply(pd.to_numeric, errors="coerce")

df["Source_Row_Index"] = np.arange(len(df))
df["Pressure_Difference"] = df["H-P1"] - df["O-P1"]

df["Pressure_Regime"] = np.select(
    [
        (df["H-P1"] >= 0) & (df["O-P1"] >= 0),
        (df["H-P1"] < 0) & (df["O-P1"] >= 0),
        (df["H-P1"] >= 0) & (df["O-P1"] < 0),
        (df["H-P1"] < 0) & (df["O-P1"] < 0),
    ],
    [
        "Both non-negative",
        "H2 negative only",
        "O2 negative only",
        "Both negative",
    ],
    default="Unclassified",
)

df["Pressure_Difference_Change"] = (
    df["Pressure_Difference"].diff()
)

df["H2_Pressure_Change"] = df["H-P1"].diff()
df["O2_Pressure_Change"] = df["O-P1"].diff()

# Identify regime transitions.
transition_indices = df.index[
    df["Pressure_Regime"].ne(df["Pressure_Regime"].shift())
].tolist()

transition_rows = []

for idx in transition_indices:
    start = max(0, idx - 5)
    end = min(len(df), idx + 6)

    for row_idx in range(start, end):
        row = df.iloc[row_idx]

        transition_rows.append({
            "Transition_Row": idx,
            "Source_Row_Index": row_idx,
            "Transition_From": (
                df.loc[idx - 1, "Pressure_Regime"]
                if idx > 0 else "Start"
            ),
            "Transition_To": df.loc[idx, "Pressure_Regime"],
            "Current_A": row["PS-I-MON"],
            "H2_Pressure": row["H-P1"],
            "O2_Pressure": row["O-P1"],
            "Pressure_Difference": row["Pressure_Difference"],
            "H2_Pressure_Change": row["H2_Pressure_Change"],
            "O2_Pressure_Change": row["O2_Pressure_Change"],
            "Pressure_Difference_Change": row[
                "Pressure_Difference_Change"
            ],
            "Temperature_C": row["T-ELY-CH1"],
            "Cell_Voltage_V": row["CV-mean"],
        })

transition_df = pd.DataFrame(transition_rows)

# Summarize the magnitude of pressure changes.
change_summary = (
    df.groupby("Pressure_Regime")
    .agg(
        Observations=("Pressure_Difference", "count"),
        Mean_Pressure_Difference=("Pressure_Difference", "mean"),
        Median_Pressure_Difference=("Pressure_Difference", "median"),
        Mean_Absolute_Difference_Change=(
            "Pressure_Difference_Change",
            lambda x: x.abs().mean()
        ),
        Maximum_Absolute_Difference_Change=(
            "Pressure_Difference_Change",
            lambda x: x.abs().max()
        ),
        Mean_Absolute_H2_Change=(
            "H2_Pressure_Change",
            lambda x: x.abs().mean()
        ),
        Mean_Absolute_O2_Change=(
            "O2_Pressure_Change",
            lambda x: x.abs().mean()
        ),
    )
    .reset_index()
)

transition_path = (
    REPORT_DIR / "norce_test4_pressure_difference_transitions.csv"
)
summary_path = (
    REPORT_DIR / "norce_test4_pressure_difference_change_summary.csv"
)

transition_df.to_csv(transition_path, index=False)
change_summary.to_csv(summary_path, index=False)

print("\nNORCE TEST4 PRESSURE DIFFERENCE TRANSITION ANALYSIS")

print("\nPRESSURE CHANGE SUMMARY")
print(change_summary.to_string(index=False))

print("\nREGIME TRANSITIONS")
print(f"Number of regime transitions: {len(transition_indices) - 1}")

print("\nTRANSITION OBSERVATIONS")
print(transition_df.to_string(index=False))

print("\nReports saved:")
print(transition_path)
print(summary_path)
