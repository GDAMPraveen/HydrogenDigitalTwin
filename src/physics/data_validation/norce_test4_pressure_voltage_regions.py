from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "data/norce/test4_subset.csv"
OUTPUT_PATH = (
    ROOT / "reports/physics/norce/"
    "norce_test4_pressure_voltage_regions.csv"
)

df = pd.read_csv(DATA_PATH)

# Define measurement-based regions.
conditions = [
    df["CV-mean"] < 0,
    df["CV-mean"] >= 0
]

labels = [
    "Negative cell voltage",
    "Non-negative cell voltage"
]

df["Voltage_Region"] = pd.Categorical(
    pd.cut(
        df["CV-mean"],
        bins=[-float("inf"), 0, float("inf")],
        labels=labels,
        right=False
    ),
    categories=labels
)

df["Current_Region"] = pd.cut(
    df["PS-I-MON"],
    bins=[-float("inf"), 0.5, 1.0, 5.0, float("inf")],
    labels=["<=0.5 A", "0.5-1 A", "1-5 A", ">5 A"],
    right=True
)

df["H2_Pressure_Negative"] = df["H-P1"] < 0
df["O2_Pressure_Negative"] = df["O-P1"] < 0

summary = (
    df.groupby(
        ["Voltage_Region", "Current_Region"],
        observed=False
    )
    .agg(
        observations=("PS-I-MON", "size"),
        negative_H2_pressure=("H2_Pressure_Negative", "sum"),
        negative_O2_pressure=("O2_Pressure_Negative", "sum"),
        mean_H2_pressure=("H-P1", "mean"),
        mean_O2_pressure=("O-P1", "mean"),
        mean_current=("PS-I-MON", "mean"),
        mean_cell_voltage=("CV-mean", "mean")
    )
    .reset_index()
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT_PATH, index=False)

print("NORCE TEST4 PRESSURE-VOLTAGE REGION ANALYSIS")
print(summary.to_string(index=False))

print(f"\nReport saved to: {OUTPUT_PATH}")