from pathlib import Path
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

ZIP_PATH = (
    ROOT
    / "data"
    / "nlr"
    / "raw"
    / "characterization"
    / "characterization_200.zip"
)

OUTPUT_DIR = ROOT / "reports" / "physics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOW_FLOW_THRESHOLD = 0.1

with zipfile.ZipFile(ZIP_PATH) as archive:
    csv_name = next(
        name for name in archive.namelist()
        if name.endswith("characterization_200.csv")
    )

    with archive.open(csv_name) as file:
        df = pd.read_csv(file)

timestamp_col = "Unnamed: 0"
flow_col = "IVAL_f_FM011_Flow"
current_col = "H2E_f_PSU_A_Current"
calc_col = "H2E_f_Elec_CalcProdRate"

df[timestamp_col] = pd.to_datetime(df[timestamp_col])

df["low_flow"] = df[flow_col] < LOW_FLOW_THRESHOLD

# Identify transitions into and out of low-flow observations.
df["flow_state_change"] = df["low_flow"].ne(df["low_flow"].shift())

transition_indices = df.index[df["flow_state_change"]].tolist()

records = []

for idx in transition_indices:
    start = max(0, idx - 5)
    end = min(len(df), idx + 6)

    for row_idx in range(start, end):
        row = df.iloc[row_idx]

        records.append({
            "transition_index": idx,
            "row_index": row_idx,
            "relative_row": row_idx - idx,
            "timestamp": row[timestamp_col],
            "low_flow": row["low_flow"],
            "measured_flow_kg_hr": row[flow_col],
            "actual_current_A": row[current_col],
            "calculated_rate_kg_hr": row[calc_col],
        })

transition_df = pd.DataFrame(records)

transition_df.to_csv(
    OUTPUT_DIR / "nlr_low_flow_transition_context.csv",
    index=False
)

print("Transition count:", len(transition_indices))
print("Transition context rows:", len(transition_df))
print("\nFirst 30 transition-context records:")
print(transition_df.head(30).to_string(index=False))

print("\nSaved:")
print(OUTPUT_DIR / "nlr_low_flow_transition_context.csv")