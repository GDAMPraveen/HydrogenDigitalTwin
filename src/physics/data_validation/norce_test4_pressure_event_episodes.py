from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = ROOT / "reports" / "physics" / "norce" / "norce_test4_pressure_event_severity.csv"
OUTPUT_DIR = ROOT / "reports" / "physics" / "norce"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROW_GAP = 5

events = pd.read_csv(INPUT_FILE)

if events.empty:
    raise ValueError("The pressure event severity report is empty.")

events = events.sort_values(
    ["Pressure_Channel", "Source_Row_Index"]
).reset_index(drop=True)

episodes = []

for channel, channel_events in events.groupby("Pressure_Channel"):
    channel_events = channel_events.sort_values("Source_Row_Index").copy()

    # Start a new episode when the gap between candidate rows exceeds 5.
    row_gaps = channel_events["Source_Row_Index"].diff()
    channel_events["Episode_Group"] = (row_gaps > MAX_ROW_GAP).cumsum()

    for group_id, group in channel_events.groupby("Episode_Group"):
        largest_idx = group["Pressure_Change_bar"].abs().idxmax()
        largest_event = group.loc[largest_idx]

        episodes.append({
            "Pressure_Channel": channel,
            "Episode_ID": f"{channel}_E{int(group_id) + 1:03d}",
            "Candidate_Count": len(group),
            "First_Candidate_Row": int(group["Source_Row_Index"].min()),
            "Last_Candidate_Row": int(group["Source_Row_Index"].max()),
            "Episode_Row_Span": int(
                group["Source_Row_Index"].max()
                - group["Source_Row_Index"].min() + 1
            ),
            "Net_Signed_Change_Sum_bar": group["Pressure_Change_bar"].sum(),
            "Largest_Absolute_Change_bar": largest_event["Pressure_Change_bar"],
            "Largest_Change_Row": int(largest_event["Source_Row_Index"]),
            "Current_Before_A": group.iloc[0]["Current_Before_A"],
            "Current_After_A": group.iloc[-1]["Current_After_A"],
            "Temperature_Before_C": group.iloc[0]["Temperature_Before_C"],
            "Temperature_After_C": group.iloc[-1]["Temperature_After_C"],
            "Cell_Voltage_Before_V": group.iloc[0]["Cell_Voltage_Before_V"],
            "Cell_Voltage_After_V": group.iloc[-1]["Cell_Voltage_After_V"],
            "Max_Absolute_Other_Pressure_Change_bar":
                group["Other_Pressure_Change_bar"].abs().max(),
        })

episode_df = pd.DataFrame(episodes)

if not episode_df.empty:
    episode_df = episode_df.sort_values(
        ["Pressure_Channel", "First_Candidate_Row"]
    )

output_file = OUTPUT_DIR / "norce_test4_pressure_event_episodes.csv"
episode_df.to_csv(output_file, index=False)

print("\nNORCE TEST4 PRESSURE EVENT EPISODES")
print(f"Grouping rule: same channel, candidate-row gap <= {MAX_ROW_GAP}")
print(f"Individual candidate changes: {len(events)}")
print(f"Grouped episodes: {len(episode_df)}")

if not episode_df.empty:
    print("\nEpisodes by pressure channel:")
    print(
        episode_df.groupby("Pressure_Channel")
        .agg(
            Episode_Count=("Episode_ID", "count"),
            Candidate_Count=("Candidate_Count", "sum"),
            Largest_Change_bar=("Largest_Absolute_Change_bar",
                                lambda x: x.abs().max()),
        )
        .to_string()
    )

    print("\nLargest episodes by candidate count:")
    print(
        episode_df.sort_values(
            "Candidate_Count", ascending=False
        ).head(20).to_string(index=False)
    )

print(f"\nReport saved:\n{output_file}")
