from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = ROOT / "reports" / "physics" / "norce" / "norce_test4_pressure_event_severity.csv"
OUTPUT_DIR = ROOT / "reports" / "physics" / "norce"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROW_GAP = 5

events = pd.read_csv(INPUT_FILE)
events = events.sort_values("Source_Row_Index").reset_index(drop=True)

if events.empty:
    raise ValueError("The pressure event severity report is empty.")

# Group candidate changes across BOTH pressure channels.
row_gaps = events["Source_Row_Index"].diff()
events["Joint_Episode_Group"] = (row_gaps > MAX_ROW_GAP).cumsum()

episodes = []

for group_id, group in events.groupby("Joint_Episode_Group"):
    largest_idx = group["Pressure_Change_bar"].abs().idxmax()
    largest_event = group.loc[largest_idx]

    channel_summary = {}

    for channel in ["H-P1", "O-P1"]:
        channel_events = group[group["Pressure_Channel"] == channel]

        channel_summary[f"{channel}_Candidate_Count"] = len(channel_events)
        channel_summary[f"{channel}_Net_Change_Sum_bar"] = (
            channel_events["Pressure_Change_bar"].sum()
            if not channel_events.empty else 0.0
        )
        channel_summary[f"{channel}_Largest_Change_bar"] = (
            channel_events.loc[
                channel_events["Pressure_Change_bar"].abs().idxmax(),
                "Pressure_Change_bar"
            ] if not channel_events.empty else None
        )

    episodes.append({
        "Joint_Episode_ID": f"J{int(group_id) + 1:03d}",
        "Channels_Involved": ", ".join(sorted(group["Pressure_Channel"].unique())),
        "Candidate_Count": len(group),
        "First_Candidate_Row": int(group["Source_Row_Index"].min()),
        "Last_Candidate_Row": int(group["Source_Row_Index"].max()),
        "Episode_Row_Span": int(
            group["Source_Row_Index"].max()
            - group["Source_Row_Index"].min() + 1
        ),
        "Largest_Absolute_Change_bar": largest_event["Pressure_Change_bar"],
        "Largest_Change_Channel": largest_event["Pressure_Channel"],
        "Largest_Change_Row": int(largest_event["Source_Row_Index"]),
        "Current_Before_A": group.iloc[0]["Current_Before_A"],
        "Current_After_A": group.iloc[-1]["Current_After_A"],
        "Temperature_Before_C": group.iloc[0]["Temperature_Before_C"],
        "Temperature_After_C": group.iloc[-1]["Temperature_After_C"],
        "Cell_Voltage_Before_V": group.iloc[0]["Cell_Voltage_Before_V"],
        "Cell_Voltage_After_V": group.iloc[-1]["Cell_Voltage_After_V"],
        **channel_summary,
    })

episode_df = pd.DataFrame(episodes).sort_values("First_Candidate_Row")

output_file = OUTPUT_DIR / "norce_test4_pressure_joint_event_episodes.csv"
episode_df.to_csv(output_file, index=False)

print("\nNORCE TEST4 JOINT PRESSURE EVENT EPISODES")
print(f"Grouping rule: any pressure-channel candidate-row gap <= {MAX_ROW_GAP}")
print(f"Individual candidates: {len(events)}")
print(f"Joint episodes: {len(episode_df)}")

print("\nEpisodes by channel participation:")
print(
    episode_df.groupby("Channels_Involved")
    .agg(
        Episode_Count=("Joint_Episode_ID", "count"),
        Candidate_Count=("Candidate_Count", "sum"),
    )
    .to_string()
)

print("\nLargest joint episodes:")
print(
    episode_df.sort_values(
        "Largest_Absolute_Change_bar",
        key=lambda s: s.abs(),
        ascending=False
    ).head(25).to_string(index=False)
)

print(f"\nReport saved:\n{output_file}")
