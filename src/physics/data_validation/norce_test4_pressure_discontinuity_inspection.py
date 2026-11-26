from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = ROOT / "data" / "norce" / "test4_subset.csv"
REPORT_DIR = ROOT / "reports" / "physics" / "norce"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

columns = [
    "PS-I-MON",
    "H-P1",
    "O-P1",
    "T-ELY-CH1",
    "CV-mean",
]

df = df[columns].apply(pd.to_numeric, errors="coerce")
df["Source_Row_Index"] = range(len(df))
df["Pressure_Difference"] = df["H-P1"] - df["O-P1"]

for col in columns:
    df[f"{col}_Change"] = df[col].diff()

events = {
    "O2_pressure_jump_row_196857": 196857,
    "O2_pressure_drop_row_196886": 196886,
    "Pressure_transition_row_14261": 14261,
    "O2_pressure_transition_row_1069": 1069,
}

output = []

for event_name, center in events.items():
    start = max(0, center - 10)
    end = min(len(df), center + 11)

    section = df.iloc[start:end].copy()
    section.insert(0, "Event", event_name)
    section["Relative_Row"] = section["Source_Row_Index"] - center

    output.append(section)

result = pd.concat(output, ignore_index=True)

output_path = (
    REPORT_DIR / "norce_test4_pressure_discontinuity_inspection.csv"
)

result.to_csv(output_path, index=False)

display_columns = [
    "Event",
    "Source_Row_Index",
    "Relative_Row",
    "PS-I-MON",
    "PS-I-MON_Change",
    "H-P1",
    "H-P1_Change",
    "O-P1",
    "O-P1_Change",
    "T-ELY-CH1",
    "T-ELY-CH1_Change",
    "CV-mean",
    "CV-mean_Change",
    "Pressure_Difference",
]

print("\nNORCE TEST4 DISCONTINUITY INSPECTION")

for event_name in events:
    print(f"\n--- {event_name} ---")

    section = result[result["Event"] == event_name]

    print(
        section[display_columns]
        .round(6)
        .to_string(index=False)
    )

print("\nReport saved:")
print(output_path)
