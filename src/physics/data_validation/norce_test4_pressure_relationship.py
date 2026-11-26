from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = ROOT / "data" / "norce" / "test4_subset.csv"
REPORT_DIR = ROOT / "reports" / "physics" / "norce"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

required = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1", "CV-mean"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required].apply(pd.to_numeric, errors="coerce")
df["Source_Row_Index"] = np.arange(len(df))

df["Pressure_Difference_H2_minus_O2"] = df["H-P1"] - df["O-P1"]

conditions = [
    (df["H-P1"] >= 0) & (df["O-P1"] >= 0),
    (df["H-P1"] < 0) & (df["O-P1"] >= 0),
    (df["H-P1"] >= 0) & (df["O-P1"] < 0),
    (df["H-P1"] < 0) & (df["O-P1"] < 0),
]

labels = [
    "Both non-negative",
    "H2 negative only",
    "O2 negative only",
    "Both negative",
]

df["Pressure_Regime"] = np.select(
    conditions,
    labels,
    default="Unclassified"
)

variables = [
    "PS-I-MON",
    "H-P1",
    "O-P1",
    "Pressure_Difference_H2_minus_O2",
    "T-ELY-CH1",
    "CV-mean",
]

summary_rows = []
correlation_rows = []

for regime, group in df.groupby("Pressure_Regime", sort=False):

    valid = group[variables].dropna()

    summary_rows.append({
        "Pressure_Regime": regime,
        "Observations": len(group),
        "Valid_Observations": len(valid),
        "Mean_Current_A": group["PS-I-MON"].mean(),
        "Mean_Cell_Voltage_V": group["CV-mean"].mean(),
        "Mean_Temperature_C": group["T-ELY-CH1"].mean(),
        "Mean_H2_Pressure": group["H-P1"].mean(),
        "Mean_O2_Pressure": group["O-P1"].mean(),
        "Mean_Pressure_Difference": group[
            "Pressure_Difference_H2_minus_O2"
        ].mean(),
        "Median_Pressure_Difference": group[
            "Pressure_Difference_H2_minus_O2"
        ].median(),
        "Min_Pressure_Difference": group[
            "Pressure_Difference_H2_minus_O2"
        ].min(),
        "Max_Pressure_Difference": group[
            "Pressure_Difference_H2_minus_O2"
        ].max(),
        "H2_O2_Pearson_Correlation": valid["H-P1"].corr(
            valid["O-P1"], method="pearson"
        ),
        "H2_O2_Spearman_Correlation": valid["H-P1"].corr(
            valid["O-P1"], method="spearman"
        ),
    })

    corr = valid.corr(method="pearson")

    for variable in variables:
        correlation_rows.append({
            "Pressure_Regime": regime,
            "Variable_1": "H-P1",
            "Variable_2": variable,
            "Pearson_Correlation": corr.loc["H-P1", variable],
        })

        correlation_rows.append({
            "Pressure_Regime": regime,
            "Variable_1": "O-P1",
            "Variable_2": variable,
            "Pearson_Correlation": corr.loc["O-P1", variable],
        })

summary = pd.DataFrame(summary_rows)
correlations = pd.DataFrame(correlation_rows)

summary_path = REPORT_DIR / "norce_test4_pressure_relationship_summary.csv"
correlation_path = REPORT_DIR / "norce_test4_pressure_correlations.csv"

summary.to_csv(summary_path, index=False)
correlations.to_csv(correlation_path, index=False)

print("\nNORCE TEST4 PRESSURE RELATIONSHIP ANALYSIS")
print("\nPRESSURE RELATIONSHIP BY REGIME")
print(summary.to_string(index=False))

print("\nPRESSURE CORRELATIONS")
print(correlations.to_string(index=False))

print("\nReports saved:")
print(summary_path)
print(correlation_path)
