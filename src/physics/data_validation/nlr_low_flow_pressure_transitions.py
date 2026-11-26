from pathlib import Path
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

ZIP_PATH = (
    ROOT / "data" / "nlr" / "raw" /
    "characterization" / "characterization_200.zip"
)

OUT = ROOT / "reports" / "physics"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.1

with zipfile.ZipFile(ZIP_PATH) as archive:
    csv_name = next(
        n for n in archive.namelist()
        if n.endswith("characterization_200.csv")
    )
    with archive.open(csv_name) as f:
        df = pd.read_csv(f)

time_col = "Unnamed: 0"
flow_col = "IVAL_f_FM011_Flow"

pressure_cols = [
    "H2E_f_PT307_Pressure",
    "H2E_f_PT312_Pressure",
    "H2E_f_PT264_Pressure",
]

df[time_col] = pd.to_datetime(df[time_col])
df["low_flow"] = df[flow_col] < THRESHOLD

starts = df.index[
    df["low_flow"] &
    ~df["low_flow"].shift(fill_value=False)
]

records = []

for episode_id, start in enumerate(starts, start=1):
    end = start

    while end + 1 < len(df) and df.loc[end + 1, "low_flow"]:
        end += 1

    for label, idx in [
        ("before", start - 1),
        ("episode_start", start),
        ("episode_end", end),
        ("after", end + 1),
    ]:
        if idx < 0 or idx >= len(df):
            continue

        row = df.iloc[idx]

        record = {
            "episode_id": episode_id,
            "position": label,
            "timestamp": row[time_col],
            "measured_flow": row[flow_col],
        }

        for col in pressure_cols:
            record[col] = row[col]

        records.append(record)

result = pd.DataFrame(records)

result.to_csv(
    OUT / "nlr_low_flow_pressure_transitions.csv",
    index=False
)

print("Episodes:", len(starts))
print("\nPressure values at episode boundaries:")
print(result.round(4).to_string(index=False))

print("\nSaved:")
print(OUT / "nlr_low_flow_pressure_transitions.csv")