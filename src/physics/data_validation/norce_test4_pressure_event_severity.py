from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = ROOT / "data" / "norce" / "test4_subset.csv"
REPORT_DIR = ROOT / "reports" / "physics" / "norce"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

columns = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1", "CV-mean"]
df = df[columns].apply(pd.to_numeric, errors="coerce")

changes = df.diff()

# Candidate pressure events: absolute one-row change >= 0.10 bar.
events = []

for pressure in ["H-P1", "O-P1"]:
    indices = changes.index[changes[pressure].abs() >= 0.10]

    for idx in indices:
        events.append({
            "Source_Row_Index": idx,
            "Pressure_Channel": pressure,
            "Pressure_Change_bar": changes.loc[idx, pressure],
            "Current_Change_A": changes.loc[idx, "PS-I-MON"],
            "Temperature_Change_C": changes.loc[idx, "T-ELY-CH1"],
            "Cell_Voltage_Change_V": changes.loc[idx, "CV-mean"],
            "Other_Pressure_Change_bar": changes.loc[
                idx, "O-P1" if pressure == "H-P1" else "H-P1"
            ],
            "Current_Before_A": df.loc[idx - 1, "PS-I-MON"] if idx > 0 else None,
            "Current_After_A": df.loc[idx, "PS-I-MON"],
            "Temperature_Before_C": df.loc[idx - 1, "T-ELY-CH1"] if idx > 0 else None,
            "Temperature_After_C": df.loc[idx, "T-ELY-CH1"],
            "Cell_Voltage_Before_V": df.loc[idx - 1, "CV-mean"] if idx > 0 else None,
            "Cell_Voltage_After_V": df.loc[idx, "CV-mean"],
        })

result = pd.DataFrame(events)

if not result.empty:
    result = result.sort_values(
        "Pressure_Change_bar",
        key=lambda s: s.abs(),
        ascending=False
    )

output_path = REPORT_DIR / "norce_test4_pressure_event_severity.csv"
result.to_csv(output_path, index=False)

print("\nNORCE TEST4 PRESSURE EVENT SEVERITY")
print(f"Candidate events (absolute change >= 0.10 bar): {len(result)}")

if not result.empty:
    print(result.to_string(index=False))

print(f"\nReport saved:\n{output_path}")
