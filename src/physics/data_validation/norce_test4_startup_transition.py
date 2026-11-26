from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "data/norce/test4_subset.csv"
OUTPUT_PATH = (
    ROOT / "reports/physics/norce/"
    "norce_test4_startup_transition.csv"
)

df = pd.read_csv(DATA_PATH)

columns = [
    "PS-I-MON",
    "H-P1",
    "O-P1",
    "T-ELY-CH1",
    "CV-mean"
]

# Preserve original row indices.
df = df[columns].copy()
df.insert(0, "Source_Row_Index", range(len(df)))

# Inspect the first 100 observations.
startup = df.head(100).copy()

# Add flags without modifying the original measurements.
startup["Negative_H2_Pressure"] = startup["H-P1"] < 0
startup["Negative_O2_Pressure"] = startup["O-P1"] < 0
startup["Negative_Cell_Voltage"] = startup["CV-mean"] < 0
startup["Low_Current"] = startup["PS-I-MON"] <= 0.5

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
startup.to_csv(OUTPUT_PATH, index=False)

print("NORCE TEST4 STARTUP TRANSITION ANALYSIS")
print("\nFirst 30 observations:")
print(startup.head(30).to_string(index=False))

print("\nFirst row with non-negative mean cell voltage:")
valid_voltage = df.index[df["CV-mean"] >= 0]

if len(valid_voltage):
    print(valid_voltage[0])
else:
    print("No non-negative voltage observations found.")

print("\nFirst row with non-negative hydrogen pressure:")
valid_pressure = df.index[df["H-P1"] >= 0]

if len(valid_pressure):
    print(valid_pressure[0])
else:
    print("No non-negative hydrogen pressure observations found.")

print("\nFirst row with current above 0.5 A:")
active_current = df.index[df["PS-I-MON"] > 0.5]

if len(active_current):
    print(active_current[0])
else:
    print("No observations above 0.5 A found.")

print(f"\nReport saved to: {OUTPUT_PATH}")