from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = ROOT / "data" / "norce" / "test4_subset.csv"
EPISODE_FILE = ROOT / "reports" / "physics" / "norce" / "norce_test4_pressure_joint_event_episodes.csv"
OUTPUT_DIR = ROOT / "reports" / "physics" / "norce"

CONTEXT_ROWS = 3

columns = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1", "CV-mean"]

df = pd.read_csv(DATA_FILE)[columns].apply(pd.to_numeric, errors="coerce")
df.index.name = "Source_Row_Index"

episodes = pd.read_csv(EPISODE_FILE)
joint = episodes[
    episodes["Channels_Involved"].str.contains("H-P1")
    & episodes["Channels_Involved"].str.contains("O-P1")
].copy()

if joint.empty:
    raise ValueError("No episodes involving both pressure channels were found.")

frames = []

for _, episode in joint.iterrows():
    start = max(0, int(episode["First_Candidate_Row"]) - CONTEXT_ROWS)
    end = min(len(df) - 1, int(episode["Last_Candidate_Row"]) + CONTEXT_ROWS)

    window = df.loc[start:end].copy()
    window.insert(0, "Joint_Episode_ID", episode["Joint_Episode_ID"])
    window["Episode_Start_Row"] = int(episode["First_Candidate_Row"])
    window["Episode_End_Row"] = int(episode["Last_Candidate_Row"])
    window["Inside_Episode"] = (
        (window.index >= episode["First_Candidate_Row"])
        & (window.index <= episode["Last_Candidate_Row"])
    )

    frames.append(window.reset_index())

result = pd.concat(frames, ignore_index=True)
result = result.sort_values(["Joint_Episode_ID", "Source_Row_Index"])

for column in columns:
    result[f"Delta_{column}"] = (
        result.groupby("Joint_Episode_ID")[column].diff()
    )

output_file = OUTPUT_DIR / "norce_test4_joint_pressure_trajectories.csv"
result.to_csv(output_file, index=False)

print("\nNORCE TEST4 JOINT PRESSURE TRAJECTORIES")
print(f"Joint episodes inspected: {len(joint)}")
print(f"Trajectory rows exported: {len(result)}")
print(f"Context rows on each side: {CONTEXT_ROWS}")

print("\nEpisode extraction summary:")
print(
    result.groupby("Joint_Episode_ID").agg(
        First_Row=("Source_Row_Index", "min"),
        Last_Row=("Source_Row_Index", "max"),
        Rows=("Source_Row_Index", "count"),
        H_P1_Min=("H-P1", "min"),
        H_P1_Max=("H-P1", "max"),
        O_P1_Min=("O-P1", "min"),
        O_P1_Max=("O-P1", "max"),
    ).to_string()
)

print(f"\nDetailed trajectories saved:\n{output_file}")
