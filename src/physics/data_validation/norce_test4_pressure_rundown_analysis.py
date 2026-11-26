from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = ROOT / "reports" / "physics" / "norce" / "norce_test4_joint_pressure_trajectories.csv"
OUTPUT_DIR = ROOT / "reports" / "physics" / "norce"

df = pd.read_csv(INPUT_FILE)

variables = [
    "H-P1", "O-P1", "PS-I-MON",
    "T-ELY-CH1", "CV-mean"
]

results = []

for episode_id, group in df.groupby("Joint_Episode_ID"):
    group = group.sort_values("Source_Row_Index")

    first = group.iloc[0]
    last = group.iloc[-1]

    result = {
        "Episode_ID": episode_id,
        "First_Row": int(first["Source_Row_Index"]),
        "Last_Row": int(last["Source_Row_Index"]),
        "Row_Count": len(group),
        "H_P1_Displacement_bar": last["H-P1"] - first["H-P1"],
        "O_P1_Displacement_bar": last["O-P1"] - first["O-P1"],
        "H_P1_Mean_Change_per_Row": group["H-P1"].diff().mean(),
        "O_P1_Mean_Change_per_Row": group["O-P1"].diff().mean(),
        "H_P1_Max_Abs_One_Row_Change": group["H-P1"].diff().abs().max(),
        "O_P1_Max_Abs_One_Row_Change": group["O-P1"].diff().abs().max(),
        "Current_Displacement_A": last["PS-I-MON"] - first["PS-I-MON"],
        "Temperature_Displacement_C": last["T-ELY-CH1"] - first["T-ELY-CH1"],
        "Cell_Voltage_Displacement_V": last["CV-mean"] - first["CV-mean"],
        "Pressure_Pearson_Correlation": group["H-P1"].corr(group["O-P1"]),
    }

    results.append(result)

summary = pd.DataFrame(results)

output_file = OUTPUT_DIR / "norce_test4_pressure_rundown_summary.csv"
summary.to_csv(output_file, index=False)

print("\nNORCE TEST4 PRESSURE RUNDOWN ANALYSIS")
print(summary.to_string(index=False))
print(f"\nReport saved:\n{output_file}")
