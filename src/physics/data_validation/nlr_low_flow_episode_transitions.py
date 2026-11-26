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
current_col = "H2E_f_PSU_A_Current"
calc_col = "H2E_f_Elec_CalcProdRate"

df[time_col] = pd.to_datetime(df[time_col])
df["low_flow"] = df[flow_col] < THRESHOLD

# Assign an episode ID to each contiguous low-flow run.
df["episode_id"] = (
    df["low_flow"] &
    (~df["low_flow"].shift(fill_value=False))
).cumsum()

low = df[df["low_flow"]].copy()

summaries = []

for episode_id, episode in low.groupby("episode_id"):
    start = episode.index.min()
    end = episode.index.max()

    before = df.iloc[max(0, start - 3):start]
    after = df.iloc[end + 1:min(len(df), end + 4)]

    summaries.append({
        "episode_id": episode_id,
        "start_time": df.loc[start, time_col],
        "end_time": df.loc[end, time_col],
        "observations": len(episode),
        "duration_seconds": len(episode),
        "mean_flow": episode[flow_col].mean(),
        "min_flow": episode[flow_col].min(),
        "max_flow": episode[flow_col].max(),
        "mean_current": episode[current_col].mean(),
        "mean_calculated_rate": episode[calc_col].mean(),
        "flow_before_mean": before[flow_col].mean(),
        "flow_after_mean": after[flow_col].mean(),
        "flow_before_last": (
            before[flow_col].iloc[-1] if len(before) else None
        ),
        "flow_after_first": (
            after[flow_col].iloc[0] if len(after) else None
        ),
    })

result = pd.DataFrame(summaries)

path = OUT / "nlr_low_flow_episode_transition_summary.csv"
result.to_csv(path, index=False)

print("Episode count:", len(result))
print("\nEpisode transition summary:")
print(result.round(4).to_string(index=False))
print("\nSaved:", path)